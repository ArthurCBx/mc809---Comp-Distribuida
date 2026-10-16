import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # TF 2.18 não roda na RTX 5070 (Blackwell): força CPU

import argparse
from client import Cliente
from server import Servidor
import flwr as fl

NCLIENTS        = 10
NTRAIN_CLIENTS  = 8   # clientes 0..7 treinam, 8..9 apenas avaliam
NROUNDS         = 20
NIID            = True
DIRICHLET_ALPHA = 0.1
FRACTION_FIT    = 0.5
LOG_DIR         = '.'

def create_client(cid):
    client = Cliente(cid, NIID, NCLIENTS, DIRICHLET_ALPHA, NTRAIN_CLIENTS, LOG_DIR)
    return client.to_client()

class Simulation():
    def __init__(self):
        self.server  = Servidor(num_clients=NCLIENTS, dirichlet_alpha=DIRICHLET_ALPHA,
                                num_train_clients=NTRAIN_CLIENTS, fraction_fit=FRACTION_FIT,
                                log_dir=LOG_DIR)

    def run_simulation(self):
        fl.simulation.start_simulation(
            client_fn     = create_client,
            num_clients   = NCLIENTS,
            config        = fl.server.ServerConfig(num_rounds=NROUNDS),
            strategy      = self.server)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulação cross-device com Flower')
    parser.add_argument('--rounds',       type=int,   default=NROUNDS)
    parser.add_argument('--alpha',        type=float, default=DIRICHLET_ALPHA)
    parser.add_argument('--fraction-fit', type=float, default=FRACTION_FIT)
    parser.add_argument('--iid',          action='store_true', help='usa IidPartitioner em vez de Dirichlet')
    parser.add_argument('--log-dir',      default=None, help='pasta dos CSVs (padrão: results/<experimento>)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # O client_fn roda nos workers do Ray, que reimportam este módulo:
    # repassamos os hiperparâmetros por variáveis de ambiente, herdadas pelos workers
    nome = f"ff{args.fraction_fit}_" + ('iid' if args.iid else f"alpha{args.alpha}")
    os.environ['SIM_ROUNDS']       = str(args.rounds)
    os.environ['SIM_NIID']         = str(int(not args.iid))
    os.environ['SIM_ALPHA']        = str(args.alpha)
    os.environ['SIM_FRACTION_FIT'] = str(args.fraction_fit)
    os.environ['SIM_LOG_DIR']      = os.path.abspath(args.log_dir or os.path.join('results', nome))

# Valores vindos da linha de comando (no processo principal e nos workers do Ray)
NROUNDS         = int(os.environ.get('SIM_ROUNDS', NROUNDS))
NIID            = bool(int(os.environ.get('SIM_NIID', int(NIID))))
DIRICHLET_ALPHA = float(os.environ.get('SIM_ALPHA', DIRICHLET_ALPHA))
FRACTION_FIT    = float(os.environ.get('SIM_FRACTION_FIT', FRACTION_FIT))
LOG_DIR         = os.environ.get('SIM_LOG_DIR', LOG_DIR)

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    for f in ('train.csv', 'evaluate.csv', 'server.csv'):
        if os.path.exists(os.path.join(LOG_DIR, f)):
            os.remove(os.path.join(LOG_DIR, f))

    print(f'NROUNDS={NROUNDS} NIID={NIID} DIRICHLET_ALPHA={DIRICHLET_ALPHA} FRACTION_FIT={FRACTION_FIT} -> {LOG_DIR}')
    Simulation().run_simulation()
