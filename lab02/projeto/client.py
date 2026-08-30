from sklearn.datasets import make_classification
import logging
import grpc
import train_eval_pb2 as pb2
import train_eval_pb2_grpc as pb2_grpc
import numpy as np
from baseline_data import baseline_data, N_SAMPLES


def change_baseline_data(baseline_data, behavior: str):
    X, y = baseline_data
    if behavior == "honest":
        return baseline_data
    elif behavior == "flip":
        for i in range(len(y)):
            if np.random.rand() < 0.5:
                y[i] = np.abs(1 - y[i])
        return X, y
    elif behavior == "random":
        noise = np.random.normal(size=X.shape)
        X = X + noise
        return X, y
    elif behavior == "single":
        noise = np.random.normal(size=N_SAMPLES)
        X[:, 0] = X[:, 0] + noise
        return X, y
    else:
        raise ValueError(f"Unknown behavior: {behavior}")


class Client(object):

    def __init__(self, client_id, behavior: str = "honest"):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.X, self.y   = change_baseline_data(baseline_data, behavior)
        self.stub        = pb2_grpc.ByzantineManagerStub(self.channel)
        self.client_id   = client_id
        self.behavior    = behavior

    def send_request(self):
        client_data = pb2.DataRequest(client_id=self.client_id, behavior=self.behavior, features=[list(x) for x in self.X], labels=list(self.y))
        return self.stub.SendData(client_data)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info("Cenario with 1 malicious client and 4 honest clients")
    client1 = Client(client_id=1, behavior="honest")
    client2 = Client(client_id=2, behavior="honest")
    client3 = Client(client_id=3, behavior="honest")
    client4 = Client(client_id=4, behavior="single")
    client5 = Client(client_id=5, behavior="honest")
    logging.info("Sending data to the server...")
    logging.info("ID 4, Behavior: single")


    logging.info("Cenario with 2 malicious clients and 3 honest clients")
    client1 = Client(client_id=1, behavior="honest")
    client2 = Client(client_id=2, behavior="flip")
    client3 = Client(client_id=3, behavior="honest")
    client4 = Client(client_id=4, behavior="honest")
    client5 = Client(client_id=5, behavior="honest")
    logging.info("ID 2, Behavior: flip")


    logging.info("Cenario with 3 malicious clients and 2 honest clients")
    client1 = Client(client_id=1, behavior="honest")
    client2 = Client(client_id=2, behavior="flip")
    client3 = Client(client_id=3, behavior="honest")
    client4 = Client(client_id=4, behavior="random")
    client5 = Client(client_id=5, behavior="honest")
    logging.info("ID 2, Behavior: flip | ID 4, Behavior: random")

