import logging
import os

import grpc
import numpy as np
import ray

import train_eval_pb2 as pb2
import train_eval_pb2_grpc as pb2_grpc
from baseline_data import baseline_data_clients, SLICE_SIZE


def change_baseline_data(baseline_data, behavior: str):
    # copia: as fatias sao views de X/y, mutar in-place corromperia os outros clientes
    X, y = baseline_data[0].copy(), baseline_data[1].copy()
    if behavior == "honest":
        return X, y
    elif behavior == "flip":
        for i in range(len(y)):
            if np.random.rand() < 0.5:
                y[i] = np.abs(1 - y[i])
        return X, y
    elif behavior == "random":
        noise = np.random.normal(size=X.shape, scale=4)
        X = X + noise
        return X, y
    elif behavior == "single":
        noise = np.random.normal(size=SLICE_SIZE, scale=4)
        X[:, 1] = X[:, 1] + noise
        return X, y
    else:
        raise ValueError(f"Unknown behavior: {behavior}")


@ray.remote
class Client:
    """Cliente como ator Ray: cada instancia roda em seu proprio processo e
    dispara a requisicao gRPC em paralelo com os demais clientes."""

    def __init__(self, client_id, behavior: str = "honest", data=None):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.X, self.y = change_baseline_data(data, behavior)
        self.stub = pb2_grpc.ByzantineManagerStub(self.channel)
        self.client_id = client_id
        self.behavior = behavior

    def send_request(self):
        client_data = pb2.DataRequest(
            client_id=self.client_id,
            behavior=self.behavior,
            features=[pb2.FeatureRow(values=[float(v) for v in x]) for x in self.X],
            labels=[int(v) for v in self.y],
        )
        response = self.stub.SendData(client_data)
        return {
            "client_id": self.client_id,
            "behavior": self.behavior,
            "isBizantine": response.isBizantine,
            "message": response.message,
        }


def run_scenario(title: str, behaviors):
    """Instancia os clientes como atores Ray e dispara TODOS em paralelo.
    So retorna quando todas as respostas do servidor chegam."""
    logging.info("=" * 70)
    logging.info(title)

    clients = [
        Client.remote(client_id=i + 1, behavior=behaviors[i], data=baseline_data_clients[i])
        for i in range(len(behaviors))
    ]

    # envio paralelo: todos os clientes disparam a requisicao ao mesmo tempo
    responses = ray.get([client.send_request.remote() for client in clients])

    for r in sorted(responses, key=lambda r: r["client_id"]):
        logging.info(f"  Client {r['client_id']:>2} ({r['behavior']:>6}): {r['message']}")

    return responses


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__))},
    )

    run_scenario(
        "Cenario com 5 clientes honestos",
        ["honest", "honest", "honest", "honest", "honest"],
    )

    run_scenario(
        "Cenario com 1 cliente malicioso e 4 honestos (ID 4: single)",
        ["honest", "honest", "honest", "single", "honest"],
    )

    run_scenario(
        "Cenario com 2 clientes maliciosos e 3 honestos (ID 2: flip | ID 4: random)",
        ["honest", "flip", "honest", "random", "honest"],
    )

    run_scenario(
        "Cenario com 3 clientes maliciosos e 2 honestos (ID 2: flip | ID 4: single | ID 5: random)",
        ["honest", "flip", "honest", "single", "random"],
    )

    ray.shutdown()
