import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import tensorflow as tf
import keras
from keras.models import Model
import time
import pandas as pd
import os
import threading

def create_server_partial_model(input_layer):
    conv1 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)
    batch_norm1 = keras.layers.BatchNormalization()(pool1)
    conv2 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(batch_norm1)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    batch_norm2 = keras.layers.BatchNormalization()(pool2)
    flatten = keras.layers.Flatten()(batch_norm2)
    dense1 = keras.layers.Dense(256, activation='relu')(flatten)
    return Model(inputs=input_layer, outputs=dense1)


class SplitLearningServer(pb2_grpc.SendActivationsServicer, pb2_grpc.SendClientGradientsServicer, pb2_grpc.TestActivationsServicer):
    def __init__(self):
        self.model = create_server_partial_model(keras.Input(shape=(6, 6, 64)))
        self.lock = threading.Lock()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.pending = {} # (client_id, batch_id) -> (tape, A1, A2)

    def SendClientActivations(self, request, context):
        """Forward do meio do modelo total."""
        A1 = tf.constant(request.activations, dtype=tf.float32)
        A1 = tf.reshape(A1, (request.batch_size, 6, 6, 64))

        with tf.GradientTape() as tape:
            tape.watch(A1)
            A2 = self.model(A1, training=True)

        self.pending[(request.client_id, request.batch_id)] = (tape, A1, A2)
        return pb2.Activations(activations=A2.numpy().flatten(), batch_size=request.batch_size, client_id=request.client_id, batch_id=request.batch_id)

    def SendGradients(self, request, context):
        """Backpropagation do meio do modelo total."""
        tape, A1, A2 = self.pending.pop((request.client_id, request.batch_id))
        dL_dA2 = tf.reshape(tf.constant(request.gradients, dtype=tf.float32), A2.shape)
        grads = tape.gradient(A2, [A1] + self.model.trainable_variables, output_gradients=dL_dA2)
        dL_dA1 = grads[0]
        with self.lock:
            self.optimizer.apply_gradients(zip(grads[1:], self.model.trainable_variables))
        return pb2.BackPropagate(gradients=dL_dA1.numpy().flatten(), batch_size=request.batch_size, client_id=request.client_id, batch_id=request.batch_id)

    def TestActivations(self, request, context):
        """Testa o envio de ativações do cliente para o servidor."""
        A1 = tf.constant(request.activations, dtype=tf.float32)
        A1 = tf.reshape(A1, (request.batch_size, 6, 6, 64))
        A2 = self.model(A1, training=False)
        return pb2.Activations(activations=A2.numpy().flatten(), batch_size=request.batch_size, client_id=request.client_id, batch_id=request.batch_id)

def serve():
    servicer = SplitLearningServer()
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    global server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    pb2_grpc.add_SendActivationsServicer_to_server(servicer, server)
    pb2_grpc.add_SendClientGradientsServicer_to_server(servicer, server)
    pb2_grpc.add_TestActivationsServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()