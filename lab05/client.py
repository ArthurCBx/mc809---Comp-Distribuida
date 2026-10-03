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
import csv
import os

MB = 2 ** 20
CSV_COLUMNS = ["client_id", "epoch", "batch", "loss", "acc", "latencia", "tempo_batch", "bytes_tx", "bytes_rx"]
TEST_CSV_COLUMNS = ["client_id", "batch", "loss", "acc", "latencia", "tempo_batch", "bytes_tx", "bytes_rx"]
TEST_SUMMARY_COLUMNS = ["client_id", "loss", "acc", "n_batches", "latencia_media", "latencia_total", "tempo_total",
                        "bytes_tx_total", "bytes_rx_total"]


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


# Métricas de comunicação calculadas como no enunciado:
#   latência = tempo da chamada gRPC (time.time antes e depois do stub)
#   bytes    = .nbytes do array de dados (float32): o tx é calculado aqui, o rx por quem recebe, a partir do tensor recebido

def send_activations_to_server(stub, activations, batch_id, batch_size, client_id):
    """Returns: (resposta, latência em s, bytes enviados)."""
    activations_list = activations.numpy().flatten()

    client_to_server_msg = pb2.Activations()
    client_to_server_msg.activations.extend(activations_list)
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id  = client_id
    client_to_server_msg.batch_id   = batch_id

    latencia_start  = time.time()
    server_response = stub.SendClientActivations(client_to_server_msg)
    latencia_end    = time.time()
    return server_response, latencia_end - latencia_start, activations_list.nbytes

def send_gradients_to_server(stub, gradients, batch_id, batch_size, client_id):
    """Returns: (resposta, latência em s, bytes enviados)."""
    gradients_list = gradients.numpy().flatten()

    backpropagate_msg = pb2.BackPropagate()
    backpropagate_msg.gradients.extend(gradients_list)
    backpropagate_msg.batch_size = batch_size
    backpropagate_msg.client_id  = client_id
    backpropagate_msg.batch_id   = batch_id

    latencia_start  = time.time()
    server_response = stub.SendGradients(backpropagate_msg)
    latencia_end    = time.time()
    return server_response, latencia_end - latencia_start, gradients_list.nbytes

def send_test_activations_to_server(stub, activations, batch_id, batch_size, client_id):
    """Forward sem treino (avaliação). Returns: (resposta, latência em s, bytes enviados)."""
    activations_list = activations.numpy().flatten()

    test_msg = pb2.Activations(activations=activations_list, batch_size=batch_size,
                               client_id=client_id, batch_id=batch_id)

    latencia_start  = time.time()
    server_response = stub.TestActivations(test_msg)
    latencia_end    = time.time()
    return server_response, latencia_end - latencia_start, activations_list.nbytes

def baseline_data():
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_step(partial_model: Model, final_model: Model, x_batch, y_batch, opt_partial, opt_final, \
               act_stub, grad_stub, acc, client_id, batch_id):

    start_batch = time.time()
    batch_size = len(x_batch)
    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)

    # 1) Forward inicial: Calcula as ativações da primeira parte do modelo
    with tf.GradientTape() as initial_tape:
        A1 = partial_model(x_batch, training=True) # shape: (B, 6, 6, 64)

    # 2) Modelo envia A1 e recebe A2 (ativações do servidor) de volta
    server_response, lat_fwd, tx_fwd = send_activations_to_server(act_stub, A1, batch_id, batch_size, client_id)
    A2 = tf.constant(server_response.activations, dtype=tf.float32)
    A2 = tf.reshape(A2, (batch_size, -1))
    rx_fwd = A2.numpy().nbytes

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
    opt_final.apply_gradients(zip(final_gradient, final_model.trainable_variables))

    # 5) Envia dL_dA2 para o servidor e recebe dL_dA1 de volta
    server_response, lat_bwd, tx_bwd = send_gradients_to_server(grad_stub, dL_dA2, batch_id, batch_size, client_id)
    dL_dA1 = tf.constant(server_response.gradients, dtype=tf.float32)
    dL_dA1 = tf.reshape(dL_dA1, A1.shape)
    rx_bwd = dL_dA1.numpy().nbytes

    # 6) Aplica a atualização dos pesos do modelo parcial usando os gradientes recebidos do servidor
    initial_gradients = initial_tape.gradient(A1, partial_model.trainable_variables, output_gradients=dL_dA1)
    opt_partial.apply_gradients(zip(initial_gradients, partial_model.trainable_variables))

    acc.update_state(y_batch, output)
    batch_acc = tf.reduce_mean(keras.metrics.sparse_categorical_accuracy(y_batch, output))

    # Métricas do batch: latência = soma das duas chamadas gRPC (rede + processamento do servidor);
    # tempo_batch = train_step inteiro, incluindo o processamento local do cliente
    return {
        "loss":        float(loss),
        "acc":         float(batch_acc),
        "latencia":    lat_fwd + lat_bwd,
        "tempo_batch": time.time() - start_batch,
        "bytes_tx":    (tx_fwd + tx_bwd) / MB,
        "bytes_rx":    (rx_fwd + rx_bwd) / MB,
    }


def train_epoch(epoch, batch_size, client_id, partial_model, final_model, opt_partial, opt_final, act_stub, grad_stub, X_train, y_train, writer):
    """Treina uma única época. O laço de épocas fica na main, que faz o FedAvg entre uma época e outra."""
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    # Permutação aleatória dos dados de treinamento para cada época
    idx = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[idx], y_train[idx]

    n_batches  = X_train.shape[0]//batch_size
    start_epoch = time.perf_counter()

    for batch in range(n_batches):
        X_batch  = X_train[batch_size * batch : batch_size * (batch+1)]
        y_batch  = y_train[batch_size * batch : batch_size * (batch+1)]

        metrics = train_step(partial_model, final_model, X_batch, y_batch, opt_partial, opt_final, act_stub, grad_stub, acc, client_id, batch)
        writer.writerow({"client_id": client_id, "epoch": epoch, "batch": batch, **metrics})

    print(f"[Cliente {client_id}] Epoch {epoch} | loss {metrics['loss']:.4f} | acc {acc.result().numpy():.4f} "
          f"| tempo {time.perf_counter() - start_epoch:.1f}s")


def fedavg(client_weights, n_samples):
    """Média dos pesos dos clientes, ponderada pela quantidade de dados de cada um.

    client_weights: uma lista por cliente, com os arrays de model.get_weights()
    n_samples:      quantidade de amostras de treino de cada cliente
    """
    total = sum(n_samples)
    return [
        sum(weights[layer] * (n / total) for weights, n in zip(client_weights, n_samples))
        for layer in range(len(client_weights[0]))
    ]


def test(partial_model, final_model, test_stub, X_test, y_test, client_id, writer):
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    loss = tf.keras.metrics.Mean()
    batch_size = 64
    n_batches  = X_test.shape[0]//batch_size
    latencia_total, bytes_tx_total, bytes_rx_total = 0.0, 0, 0
    start_test = time.time()

    for batch in range(n_batches):
        start_batch = time.time()
        X_batch  = X_test[batch_size * batch : batch_size * (batch+1)]
        y_batch  = y_test[batch_size * batch : batch_size * (batch+1)]

        x_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        A1 = partial_model(x_batch, training=False)

        # No teste há só o forward: uma chamada gRPC (A1 vai, A2 volta)
        server_response, latencia, bytes_tx = send_test_activations_to_server(test_stub, A1, batch, batch_size, client_id)
        A2 = tf.constant(server_response.activations, dtype=tf.float32)
        A2 = tf.reshape(A2, (batch_size, -1))
        bytes_rx = A2.numpy().nbytes

        output = final_model(A2, training=False)
        batch_loss = keras.losses.sparse_categorical_crossentropy(y_batch, output)
        acc.update_state(y_batch, output)
        loss.update_state(batch_loss)

        writer.writerow({
            "client_id":   client_id,
            "batch":       batch,
            "loss":        float(tf.reduce_mean(batch_loss)),
            "acc":         float(tf.reduce_mean(keras.metrics.sparse_categorical_accuracy(y_batch, output))),
            "latencia":    latencia,
            "tempo_batch": time.time() - start_batch,
            "bytes_tx":    bytes_tx / MB,
            "bytes_rx":    bytes_rx / MB,
        })
        latencia_total += latencia
        bytes_tx_total += bytes_tx
        bytes_rx_total += bytes_rx

    print(f"[Cliente {client_id}] Test loss: {loss.result().numpy():.4f} | Test accuracy: {acc.result().numpy():.4f}")
    # Resumo do teste do cliente: médias por batch e totais da avaliação inteira
    return {
        "client_id":      client_id,
        "loss":           float(loss.result()),
        "acc":            float(acc.result()),
        "n_batches":      n_batches,
        "latencia_media": latencia_total / n_batches,
        "latencia_total": latencia_total,
        "tempo_total":    time.time() - start_test,
        "bytes_tx_total": bytes_tx_total / MB,
        "bytes_rx_total": bytes_rx_total / MB,
    }


# Os clientes rodam na CPU: a GPU fica só para o servidor. Se os actors enxergarem a GPU, cada processo
# TensorFlow tenta reservá-la e a disputa com o servidor deixa cada batch ~50x mais lento
@ray.remote(runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": ""}})
class Client:
    def __init__(self, client_id, X_train, y_train, results_dir):
        self.client_id = client_id
        self.results_path = os.path.join(results_dir, f"results_client_{client_id}.csv")
        self.test_results_path = os.path.join(results_dir, f"results_test_client_{client_id}.csv")
        self.partial_model = create_partial_model(keras.Input(shape=(32, 32, 3)))
        self.final_model = create_final_model(keras.Input(shape=(256,)))
        self.opt_partial = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.opt_final = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.X_train, self.y_train = X_train, y_train
        MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
        channel = grpc.insecure_channel('localhost:50051', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
        self.act_stub   = pb2_grpc.SendActivationsStub(channel)
        self.final_stub = pb2_grpc.SendClientGradientsStub(channel)
        self.test_stub  = pb2_grpc.TestActivationsStub(channel)

        # Um CSV por cliente: cada actor é um processo separado, e escrever todos no mesmo arquivo misturaria as linhas
        self.results_file = open(self.results_path, "w", newline="")
        self.writer = csv.DictWriter(self.results_file, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()

    def train_epoch(self, epoch, batch_size):
        train_epoch(epoch, batch_size, self.client_id, self.partial_model,
                    self.final_model, self.opt_partial, self.opt_final, self.act_stub,
                    self.final_stub, self.X_train, self.y_train, self.writer)
        self.results_file.flush()  # garante os dados da época no disco mesmo se o treino for interrompido

    def num_samples(self):
        return len(self.X_train)

    def get_weights(self):
        """Pesos de M1 e M3. Inclui as médias móveis da BatchNorm, que também entram na média."""
        return self.partial_model.get_weights(), self.final_model.get_weights()

    def set_weights(self, partial_weights, final_weights):
        self.partial_model.set_weights(partial_weights)
        self.final_model.set_weights(final_weights)

    def test(self, X_test, y_test):
        with open(self.test_results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TEST_CSV_COLUMNS)
            writer.writeheader()
            return test(self.partial_model, self.final_model, self.test_stub, X_test, y_test, self.client_id, writer)


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

    # Caminho absoluto: os actors do Ray rodam em outros processos, que não necessariamente têm o mesmo diretório de trabalho
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    clients = [Client.remote(i, X, y, results_dir) for i, (X, y) in enumerate(zip(X_train, y_train))]
    del X_train, y_train # libera memória
    n_samples = ray.get([client.num_samples.remote() for client in clients])

    # FedAvg parte de um modelo global único: todos os clientes começam com os pesos do cliente 0
    partial_weights, final_weights = ray.get(clients[0].get_weights.remote())
    ray.get([client.set_weights.remote(partial_weights, final_weights) for client in clients])

    for epoch in range(10):
        # 1) Cada cliente treina uma época com os próprios dados, em paralelo
        ray.get([client.train_epoch.remote(epoch, batch_size=64) for client in clients])

        # 2) Agregação: média ponderada dos pesos de M1 e M3 de todos os clientes
        client_weights = ray.get([client.get_weights.remote() for client in clients])
        partial_weights = fedavg([w[0] for w in client_weights], n_samples)
        final_weights   = fedavg([w[1] for w in client_weights], n_samples)

        # 3) Distribuição: todos os clientes recebem o modelo global para a próxima época
        ray.get([client.set_weights.remote(partial_weights, final_weights) for client in clients])
        print(f"[FedAvg] Epoch {epoch} | pesos agregados de {len(clients)} clientes")

    # Após o último FedAvg todos os clientes têm o mesmo M1/M3: as partições de teste somadas avaliam o modelo global
    test_results = ray.get([client.test.remote(X_test[i], y_test[i]) for i, client in enumerate(clients)])

    with open(os.path.join(results_dir, "test_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(test_results)
