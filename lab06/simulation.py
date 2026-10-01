from client import Cliente
from server import Servidor
import flwr as fl

NCLIENTS        = 10
NROUNDS         = 20
NIID            = True
DIRICHLET_ALPHA = 0.1
FRACTION_FIT    = 0.2

def create_client(cid):
    client = Cliente(cid, NIID, NCLIENTS, DIRICHLET_ALPHA)
    return client.to_client()

class Simulation():
    def __init__(self):
        self.server  = Servidor(num_clients=NCLIENTS, dirichlet_alpha=DIRICHLET_ALPHA, fraction_fit=FRACTION_FIT)

    def run_simulation(self):
        fl.simulation.start_simulation(
            client_fn     = create_client,
            num_clients   = NCLIENTS,
            config        = fl.server.ServerConfig(num_rounds=NROUNDS),
            strategy      = self.server)