import copy
import logging
import os
from concurrent import futures

import grpc
import numpy as np
import ray
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import train_eval_pb2 as pb2
import train_eval_pb2_grpc as pb2_grpc

from baseline_data import X_base, y_base, X_test, y_test, SLICE_SIZE


@ray.remote
class Aggregator:
    """Guarda o modelo base do servidor e julga cada lote recebido.

    O servidor treina o detector em X_base/y_base, que nenhum cliente conhece.
    Como todos os clientes deveriam amostrar da mesma distribuicao, o servidor
    pode julgar se cada lote recebido é consistente com o baseline.

    O treinamento do modelo federado (self.model) só acontece DEPOIS que todos
    os clientes esperados enviam seus dados: cada lote honesto é acumulado e, ao
    chegar o ultimo cliente, o modelo é treinado de uma vez com todos eles.
    """

    def __init__(self, expected_clients=5, batch_size=SLICE_SIZE):
        self.expected_clients = expected_clients
        self.batch_size = batch_size

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_base, y_base, test_size=0.5, random_state=42
        )

        # detector: modelo de referencia usado apenas para decidir se um cliente
        # é bizantino. Fica fixo durante toda a rodada, para julgar todos os
        # clientes com o mesmo criterio.
        self.detector = SGDClassifier(max_iter=200)
        self.detector.fit(X_fit, y_fit)

        score_on_unknown = self.detector.score(X_cal, y_cal)
        self.score_threshold = 0.45 * (score_on_unknown + self.detector.score(X_fit, y_fit))

        # modelo federado, treinado com os dados dos clientes honestos.
        self.model = None
        self._reset_round()

    def _reset_round(self):
        self._pending_X = []
        self._pending_y = []
        self._received = 0

    def _train_round(self):
        """Chamado quando todos os clientes ja enviaram seus dados."""
        if self._pending_X:
            X_all = np.vstack(self._pending_X)
            y_all = np.concatenate(self._pending_y)
            self.model = copy.deepcopy(self.detector)
            self.model.partial_fit(X_all, y_all)
            summary = (
                f"All {self.expected_clients} clients received. Model trained with "
                f"{len(self._pending_X)} honest client(s) ({len(y_all)} samples).\n"
                f"Final model score on test data: {self.eval_final_model():.4f}"
            )
        else:
            self.model = None
            summary = (
                f"All {self.expected_clients} clients received, but none were "
                f"classified as honest. No training performed."
            )
        self._reset_round()
        return summary

    def eval_final_model(self): 
        """Avalia o modelo federado final com os dados de teste."""
        if self.model is None:
            return "No model to evaluate."
        return self.model.score(X_test, y_test)

    def evaluate(self, client_id, features, labels):
        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)

        score = float(self.detector.score(X, y))
        if score < self.score_threshold:
            is_byzantine = True
            message = (
                f"Client {client_id} was defined as a Byzantine client. "
                f"Its data will not be used to train the model."
            )
        else:
            is_byzantine = False
            message = (
                f"Client {client_id} was classified as honest. "
                f"Its data will be used to train the model."
            )
            self._pending_X.append(X)
            self._pending_y.append(y)

        self._received += 1

        result = {"isBizantine": is_byzantine, "message": message}
        # o treinamento so acontece depois de receber os dados de TODOS os clientes
        if self._received >= self.expected_clients:
            result["training_summary"] = self._train_round()

        return result


class ByzantineServiceManager(pb2_grpc.ByzantineManagerServicer):
    def __init__(self, expected_clients=5):
        self.aggregator = Aggregator.remote(expected_clients=expected_clients)

    def SendData(self, request, context):
        features = [list(row.values) for row in request.features]
        labels = list(request.labels)

        if not features or len(features) != len(labels):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("features e labels devem ter o mesmo tamanho e nao ser vazios")
            return pb2.DataResponse()

        result = ray.get(
            self.aggregator.evaluate.remote(request.client_id, features, labels)
        )

        logging.info(result["message"])

        training_summary = result.pop("training_summary", None)
        if training_summary:
            logging.info(training_summary)

        return pb2.DataResponse(
            isBizantine=result["isBizantine"],
            message=result["message"],
        )


def serve():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__))},
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    pb2_grpc.add_ByzantineManagerServicer_to_server(ByzantineServiceManager(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("Server started at 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
