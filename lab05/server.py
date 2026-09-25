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

def create_server_partial_model(input_layer):
    conv1 = keras.layers.Conv2D(128, (3, 3), activation='relu')(input_layer)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)
    batch_norm1 = keras.layers.BatchNormalization()(pool1)
    conv2 = keras.layers.Conv2D(256, (3, 3), activation='relu')(batch_norm1)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    batch_norm2 = keras.layers.BatchNormalization()(pool2)
    flatten = keras.layers.Flatten()(batch_norm2)
    dense1 = keras.layers.Dense(256, activation='relu')(flatten)
    return Model(inputs=input_layer, outputs=dense1)


def get_partial_activations_and_forward_pass(partial_model, partial_activations):
    """Get the activations of the server model for a given input partial_activations.
    
    Returns: The activations and the gradient tape."""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(partial_activations)
        activations = partial_model(partial_activations)
    return activations, tape


class ServerModel(tf.keras.models.Model):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.partial_model = create_server_partial_model((64,))

    def forward_pass(self, partial_activations):
        return get_partial_activations_and_forward_pass(self.partial_model, partial_activations)


class Server(pb2_grpc.ServerToClientServicer):

    def __init__(self):
        self.server_model = ServerModel().partial_model

    def SendServerActivations(self, request, context):
        activations = tf.convert_to_tensor(request.activations, dtype=tf.float32)
        # -1 faz com que a dimensao seja calculada automaticamente
        activations = tf.reshape(activations, (request.batch_size, -1)) 

        activations, tape = get_partial_activations_and_forward_pass(self.server_model, activations)

        backpropagate_obj = pb2.SendServerActivations(activations=activations.numpy().flatten(),
                                  batch_size=request.batch_size,
                                  client_id=request.client_id)

        


        # print("BACKWARD")
        # server_gradients = tape.gradient(loss, self.server_model.trainable_variables)
        # self.optimizer.apply_gradients(zip(server_gradients, self.server_model.trainable_variables))

        # activations_gradients = tape.gradient(loss, activations)
        # response              = pb2.ServerToClient()

        # response.gradients.extend(activations_gradients.numpy().flatten())

        # return response


def serve():
    global server
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    pb2_grpc.add_ServerToClientServicer_to_server(ServerToClient(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()