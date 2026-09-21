import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import fashion_mnist

import grpc
import data_payload_pb2 as pb2
import data_payload_pb2_grpc as pb2_grpc


class Client:
    def __init__(self, model, client_id):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub = pb2_grpc.SimilarityServiceStub(self.channel)
        self.model = model
        self.client_id = client_id

    def forward(self, input_data):
        return self.model(input_data, training=False)

    def send_output_vector(self, input_data, pair_id, id):
        output_vector = self.forward(input_data)

        # Convert the output vector to a list of floats
        output_vector_list = output_vector.numpy().ravel().tolist()
        data_payload = pb2.DataPayload(output_vector=output_vector_list, 
                                       pair_id=pair_id, 
                                       client_id=self.client_id,
                                       class_id=id)
        self.stub.CompareData(data_payload)
        return data_payload

if __name__ == "__main__":
    SAMPLES=5

    base_model = keras.models.load_model("fashion_mnist_model.keras")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_test = X_test / 255.0

    client_1 = Client(base_model, client_id=1)
    client_2 = Client(base_model, client_id=2)

    X_test = X_test.reshape(-1, 28, 28, 1)

    for i in range(SAMPLES):
        idx_1 = np.random.randint(0, len(X_test))
        input_data_1 = X_test[idx_1]
        class_id_1 = y_test[idx_1]

        idx_2 = np.random.randint(0, len(X_test))
        input_data_2 = X_test[idx_2]
        class_id_2 = y_test[idx_2]

        output_payload_1 = client_1.send_output_vector(np.expand_dims(input_data_1, axis=0), pair_id=i, id=class_id_1)
        output_payload_2 = client_2.send_output_vector(np.expand_dims(input_data_2, axis=0), pair_id=i, id=class_id_2)



