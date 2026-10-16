import os
import flwr as fl
import tensorflow as tf
from flwr.common import FitIns, EvaluateIns, GetPropertiesIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.criterion import Criterion
from flwr.server.strategy.aggregate import aggregate
from flwr_datasets import FederatedDataset 
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions, plot_comparison_label_distribution

class GrupoCriterion(Criterion):
    """Seleciona apenas clientes de treino (is_train=True) ou apenas de avaliação (is_train=False)."""
    def __init__(self, cid_map, num_train_clients, is_train):
        self.cid_map           = cid_map
        self.num_train_clients = num_train_clients
        self.is_train          = is_train

    def select(self, client):
        return (self.cid_map[client.cid] < self.num_train_clients) == self.is_train


class Servidor(fl.server.strategy.FedAvg):
    def __init__(self, num_clients, dirichlet_alpha, num_train_clients, fraction_fit=0.2, fraction_evaluate=1.0, log_dir='.'):
        self.num_clients       = num_clients
        self.dirichlet_alpha   = dirichlet_alpha
        self.num_train_clients = num_train_clients
        self.num_eval_clients  = num_clients - num_train_clients
        self.log_dir           = log_dir

        # Mapeia o id real do ClientProxy (node_id) para o id lógico do cliente (0..num_clients-1)
        self.cid_map = {}

        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                         min_available_clients=num_clients)

    def map_clients(self, client_manager):
        client_manager.wait_for(self.num_clients)
        for proxy_cid, proxy in client_manager.all().items():
            if proxy_cid not in self.cid_map:
                res = proxy.get_properties(GetPropertiesIns({}), timeout=None, group_id=0)
                self.cid_map[proxy_cid] = int(res.properties['cid'])

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""

        config = {
            'server_round': server_round,
        } 
        fit_ins = FitIns(parameters, config)

        self.map_clients(client_manager)

        # A fração é aplicada apenas sobre o grupo de clientes de treino
        sample_size = max(int(self.fraction_fit * self.num_train_clients), 1)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.num_clients,
            criterion=GrupoCriterion(self.cid_map, self.num_train_clients, is_train=True)
        )

        # Return client/config pairs
        print(clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):       
        parameters_list = []
        for _, fit_res in results:
            parameters = parameters_to_ndarrays(fit_res.parameters)
            exemplos   = int(fit_res.num_examples)

            parameters_list.append([parameters, exemplos])

        agg_parameters = aggregate(parameters_list)
        agg_parameters = ndarrays_to_parameters(agg_parameters)

        return agg_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {
            'server_round': server_round,
        } 

        evaluate_ins = EvaluateIns(parameters, config)


        self.map_clients(client_manager)

        # A fração é aplicada apenas sobre o grupo de clientes de avaliação
        sample_size = max(int(self.fraction_evaluate * self.num_eval_clients), 1)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.num_clients,
            criterion=GrupoCriterion(self.cid_map, self.num_train_clients, is_train=False)
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        accuracies = []

        for _, response in results:
            acc = response.metrics['accuracy']
            accuracies.append(acc)

        avg_acc = sum(accuracies)/len(accuracies)
        print(f"Rodada {server_round} acurácia agregada: {avg_acc}")
        with open(os.path.join(self.log_dir, 'server.csv'), 'a') as file:
            file.write(f'{server_round}, {avg_acc}\n')

        return avg_acc, {}