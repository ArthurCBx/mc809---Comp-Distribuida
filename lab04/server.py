import numpy as np
from tensorflow import keras
import grpc
import data_payload_pb2 as pb2
import data_payload_pb2_grpc as pb2_grpc
import logging
import threading
from concurrent import futures
from google.protobuf import empty_pb2
from keras_distance import euclidean_distance

EXPECTED_CLIENTS = 2
THRESHOLD = 0.5


class SimilarityService(pb2_grpc.SimilarityServiceServicer):
    def __init__(self, threshold=THRESHOLD):
        self.threshold = threshold
        self.pending = {}
        self.lock = threading.Lock()
        self.model = keras.models.load_model('fashion_mnist_siamese_model.keras', custom_objects={'euclidean_distance': euclidean_distance})

    def CompareData(self, request, context):
        vector = np.array(request.output_vector, dtype=np.float32)

        with self.lock:
            slot = self.pending.setdefault(request.pair_id, {})
            slot[request.client_id] = (vector, request.class_id)

            if len(slot) < EXPECTED_CLIENTS:
                return empty_pb2.Empty()

            pair = self.pending.pop(request.pair_id)
            (id_a, (vector_a, class_a)), (id_b, (vector_b, class_b)) = sorted(pair.items())

        distance = float(self.model([vector_a.reshape(1, -1), vector_b.reshape(1, -1)])[0, 0])
        is_similar = distance < self.threshold
        if is_similar:
            message = f"Clients {id_a} and {id_b} are similar (distance: {distance:.4f})."
        else:
            message = f"Clients {id_a} and {id_b} are not similar (distance: {distance:.4f})."
        logging.info(message)
        logging.warning(f"Correct answer is: {'similar' if class_a == class_b else 'not similar'}\n")
        return empty_pb2.Empty()

def serve():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    pb2_grpc.add_SimilarityServiceServicer_to_server(SimilarityService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("Server started at 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()