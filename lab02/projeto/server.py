import grpc
from concurrent import futures
from sklearn.linear_model import LogisticRegression

import train_eval_pb2 as pb2
import train_eval_pb2_grpc as pb2_grpc
import ray

from baseline_data import baseline_data, N_SAMPLES

ray.init()

@ray.remote
class Aggregator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class ByzantineServiceManager(pb2_grpc.ByzantineManager):
    def __init__(self):
        self.aggregator = Aggregator.remote(expected_clients=5)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    pb2_grpc.add_TrainEvalServicer_to_server(TrainEvalServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
