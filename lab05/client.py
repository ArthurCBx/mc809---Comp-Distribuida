import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf
import keras
from keras.models import Model
import grpc
import time
import numpy as np
import os

def create_partial_model(input_layer):
    layer1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    layer2 = keras.layers.MaxPooling2D((2, 2))(layer1)
    batch_norm1 = keras.layers.BatchNormalization()(layer2)
    layer3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(batch_norm1)
    layer4 = keras.layers.MaxPooling2D((2, 2))(layer3)
    batch_norm2 = keras.layers.BatchNormalization()(layer4)
    return Model(inputs=input_layer, outputs=layer4)


def create_final_model(input_layer):
    dense1 = keras.layers.Dense(128, activation='relu')(input_layer)
    dense2 = keras.layers.Dense(10, activation='softmax')(dense1)
    return Model(inputs=input_layer, outputs=dense2)


def get_activations(model, input_data):
    """Get the activations of a model for a given input_data (can be partial activations).
    
    Returns: The activations and the gradient tape."""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_data)
        activations = model(input_data)
    return activations, tape

def send_activations_to_server(stub, activations, labels, batch_size, client_id):
    activations_list = activations.numpy().flatten()

    client_to_server_msg = pb2.ClientToServer()
    client_to_server_msg.activations.extend(activations_list)
    client_to_server_msg.labels.extend(labels.flatten())
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id = client_id

    server_response = stub.SendClientActivations(client_to_server_msg)
    return server_response

def baseline_data():
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_step(model, x_batch, y_batch, batch, optimizer, epoch, stub):

    activations, tape     = get_activations(model, x_batch)
    flattened_activations = tf.reshape(activations, (activations.shape[0], -1))

    latencia_start  = time.time()
    server_response = send_activations_to_server(stub, flattened_activations, y_batch, len(x_batch), 1)
    latencia_end    = time.time()

    print("Received response from server")
    activations_grad = tf.convert_to_tensor(server_response.gradients, dtype=tf.float32)
    activations_grad = tf.reshape(activations_grad, activations.shape)

    client_gradient = tape.gradient(
        activations,
        model.trainable_variables,
        output_gradients=activations_grad
    )

    bytes_tx  = flattened_activations.numpy().nbytes
    bytes_rx  = activations_grad.numpy().nbytes
    latencia  = latencia_end - latencia_start
    loss      = server_response.loss
    acc       = server_response.acc

    print(f"Latencia: {latencia} segundos")
    print(f"Data Tx: {bytes_tx / 2**20} MB")
    print(f"Data Rx: {bytes_rx / 2**20} MB")

    optimizer.apply_gradients(zip(client_gradient, model.trainable_variables))

    with open('results.csv', 'a') as f:
        f.write(f"{epoch}, {batch}, {loss}, {acc}, {latencia}, {bytes_tx / 2**20}, {bytes_rx / 2**20}\n")


def train():
    for epoch in range(10):
    batch_size = 64
    n_batches  = X_train.shape[0]//batch_size

    for batch in range(n_batches):
        print(f"Epoch {epoch} - Batch {batch}/{n_batches}")

        X_batch  = X_train[batch_size * batch : batch_size * (batch+1)]
        y_batch  = y_train[batch_size * batch : batch_size * (batch+1)]

        train_step(partial_model, X_batch, y_batch, batch, client_optimizer, epoch, stub)


if __name__ == "__main__":
    X_train, y_train = baseline_data()[0]
    X_test, y_test = baseline_data()[1]

    partial_model    = create_partial_model(keras.Input(shape=(32, 32, 3)))
    client_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    channel = grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])
    stub = pb2_grpc.SplitLearningStub(channel)