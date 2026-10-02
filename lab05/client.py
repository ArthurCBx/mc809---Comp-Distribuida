import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf
import keras
from keras.models import Model
import grpc
import time
import numpy as np
import ray
import threading


def create_partial_model(input_layer):
    layer1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    layer2 = keras.layers.MaxPooling2D((2, 2))(layer1)
    batch_norm1 = keras.layers.BatchNormalization()(layer2)
    layer3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(batch_norm1)
    layer4 = keras.layers.MaxPooling2D((2, 2))(layer3)
    batch_norm2 = keras.layers.BatchNormalization()(layer4)
    return Model(inputs=input_layer, outputs=batch_norm2)


def create_final_model(input_layer):
    dense1 = keras.layers.Dense(128, activation='relu')(input_layer)
    dense2 = keras.layers.Dense(10, activation='softmax')(dense1)
    return Model(inputs=input_layer, outputs=dense2)


def send_activations_to_server(stub, activations, batch_id, batch_size, client_id):
    activations_list = activations.numpy().flatten()

    client_to_server_msg = pb2.Activations()
    client_to_server_msg.activations.extend(activations_list)
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id  = client_id
    client_to_server_msg.batch_id   = batch_id

    server_response = stub.SendClientActivations(client_to_server_msg)
    return server_response

def send_gradients_to_server(stub, gradients, batch_id, batch_size, client_id):
    gradients_list = gradients.numpy().flatten()

    backpropagate_msg = pb2.BackPropagate()
    backpropagate_msg.gradients.extend(gradients_list)
    backpropagate_msg.batch_size = batch_size
    backpropagate_msg.client_id  = client_id
    backpropagate_msg.batch_id   = batch_id

    server_response = stub.SendGradients(backpropagate_msg)
    return server_response

def baseline_data():
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train.astype(np.float32), y_train), (x_test.astype(np.float32), y_test)


def train_step(partial_model: Model, final_model: Model, x_batch, y_batch, opt_partial, opt_final, \
               act_stub, grad_stub, acc, client_id, batch_id):

    batch_size = len(x_batch)
    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    lock = threading.Lock()
    
    # 1) Forward inicial: Calcula as ativações da primeira parte do modelo
    with tf.GradientTape() as initial_tape:
        A1 = partial_model(x_batch, training=True) # shape: (B, 6, 6, 64)

    # 2) Modelo envia A1 e recebe A2 (ativações do servidor) de volta
    server_response = send_activations_to_server(act_stub, A1, batch_id, batch_size, client_id)
    A2 = tf.constant(server_response.activations, dtype=tf.float32)
    A2 = tf.reshape(A2, (batch_size, -1))

    # 3) Forward da parte final do modelo e cálculo da loss
    with tf.GradientTape() as final_tape:
        final_tape.watch(A2)
        output = final_model(A2, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y_pred=output, y_true=y_batch)
        loss = tf.reduce_mean(loss)

    # 4) Backward: Calcula os gradientes da loss em relação aos pesos do modelo final e aplica a atualização
    final_gradient = final_tape.gradient(
        loss,
        [A2] + final_model.trainable_variables,
    )
    dL_dA2 = final_gradient[0]
    final_gradient = final_gradient[1:]
    with lock:
        opt_final.apply_gradients(zip(final_gradient, final_model.trainable_variables))

    # 5) Envia dL_dA2 para o servidor e recebe dL_dA1 de volta
    server_response = send_gradients_to_server(grad_stub, dL_dA2, batch_id, batch_size, client_id)
    dL_dA1 = tf.constant(server_response.gradients, dtype=tf.float32)
    dL_dA1 = tf.reshape(dL_dA1, A1.shape)

    # 6) Aplica a atualização dos pesos do modelo parcial usando os gradientes recebidos do servidor
    initial_gradients = initial_tape.gradient(A1, partial_model.trainable_variables, output_gradients=dL_dA1)
    with lock:
        opt_partial.apply_gradients(zip(initial_gradients, partial_model.trainable_variables))

    acc.update_state(y_batch, output)
    return float(loss)


@ray.remote
def train(num_epochs, batch_size, client_id, partial_model, final_model, opt_partial, opt_final, act_stub, grad_stub, X_train, y_train):
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in range(num_epochs):
        # Permutação aleatória dos dados de treinamento para cada época
        idx = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[idx], y_train[idx]

        n_batches  = X_train.shape[0]//batch_size

        for batch in range(n_batches):
            X_batch  = X_train[batch_size * batch : batch_size * (batch+1)]
            y_batch  = y_train[batch_size * batch : batch_size * (batch+1)]

            loss = train_step(partial_model, final_model, X_batch, y_batch, opt_partial, opt_final, act_stub, grad_stub, acc, client_id, batch)

            print(f"Epoch {epoch} - Batch {batch}/{n_batches} | loss {loss}")
        print(f"Epoch {epoch} | loss {loss} | acc {acc.result().numpy()}")
        acc.reset_state()


@ray.remote
def test(partial_model, final_model, test_stub, X_test, y_test):
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    batch_size = 64
    n_batches  = X_test.shape[0]//batch_size

    for batch in range(n_batches):
        X_batch  = X_test[batch_size * batch : batch_size * (batch+1)]
        y_batch  = y_test[batch_size * batch : batch_size * (batch+1)]

        x_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        A1 = partial_model(x_batch, training=False)

        server_response = test_stub.TestActivations(pb2.Activations(
            activations=A1.numpy().flatten(),
            batch_size=batch_size,
            client_id=0,
            batch_id=batch
        ))
        A2 = tf.constant(server_response.activations, dtype=tf.float32)
        A2 = tf.reshape(A2, (batch_size, -1))

        output = final_model(A2, training=False)
        acc.update_state(y_batch, output)

    print(f"Test accuracy: {acc.result().numpy()}")

if __name__ == "__main__":
    NUM_CLIENTS = 3
    ray.init(ignore_reinit_error=True)

    dataset = baseline_data()
    X_train, y_train = dataset[0]
    X_test, y_test = dataset[1]

    X_train = np.array_split(X_train, NUM_CLIENTS)
    y_train = np.array_split(y_train, NUM_CLIENTS)
    X_test  = np.array_split(X_test, NUM_CLIENTS)
    y_test  = np.array_split(y_test, NUM_CLIENTS)

    partial_model = create_partial_model(keras.Input(shape=(32, 32, 3)))
    final_model   = create_final_model(keras.Input(shape=(256,)))
    opt_partial   = tf.keras.optimizers.Adam(learning_rate=1e-3)
    opt_final   = tf.keras.optimizers.Adam(learning_rate=1e-3)

    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    channel = grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])
    act_stub   = pb2_grpc.SendActivationsStub(channel)
    final_stub = pb2_grpc.SendClientGradientsStub(channel)
    test_stub  = pb2_grpc.TestActivationsStub(channel)
    for i, data in enumerate(zip(X_train, y_train)):
        train.remote(10, 64, i, partial_model, final_model, opt_partial, opt_final, act_stub, final_stub, data[0], data[1])
