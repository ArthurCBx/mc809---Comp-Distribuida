import os
import flwr as fl
import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions, plot_comparison_label_distribution


class Cliente(fl.client.NumPyClient):
    def __init__(self, cid, niid, num_clients, dirichlet_alpha, num_train_clients, log_dir='.'):
        self.cid               = int(cid)
        self.niid              = niid
        self.num_clients       = num_clients
        self.dirichlet_alpha   = dirichlet_alpha
        self.num_train_clients = num_train_clients
        self.log_dir           = log_dir

        # Cross-device: clientes [0, num_train_clients) só treinam e os demais só avaliam
        self.is_train_client = self.cid < num_train_clients

        self.x, self.y = self.load_data()
        self.model     = self.create_model(self.x.shape)

    def get_parameters(self, config):
        return self.model.get_weights()

    def get_properties(self, config):
        # Permite ao servidor mapear o id do ClientProxy (node_id) para o id lógico do cliente
        return {'cid': self.cid, 'is_train_client': self.is_train_client}

    def load_data(self):
        # Clientes de treino particionam apenas o split "train" e clientes de avaliação apenas o "test"
        if self.is_train_client:
            split, num_partitions, partition_id = 'train', self.num_train_clients, self.cid
        else:
            split, num_partitions, partition_id = 'test', self.num_clients - self.num_train_clients, self.cid - self.num_train_clients

        if self.niid:
            partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=0,
                                    self_balancing=False)
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)

        fds  = FederatedDataset(dataset='ylecun/mnist', partitioners={split: partitioner})
        data = fds.load_partition(partition_id, split).with_format("numpy")

        return np.asarray(data['image'])/255.0, np.asarray(data['label'])

    def create_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16,  activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),

        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, parameters, config):
        assert self.is_train_client, f'Cliente {self.cid} é de avaliação e não deveria treinar'
        self.model.set_weights(parameters)

        history = self.model.fit(self.x, self.y, epochs=1)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()

        fit_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss,
        }

        self.log_client('train.csv', config['server_round'], acc, loss)
        return trained_parameters, len(self.x), fit_msg

    def evaluate(self, parameters, config):
        assert not self.is_train_client, f'Cliente {self.cid} é de treino e não deveria avaliar'
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x, self.y)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        self.log_client('evaluate.csv', config['server_round'], acc, loss)
        return loss, len(self.x), eval_msg

    def log_client(self, file_name, server_round, acc, loss):
        with open(os.path.join(self.log_dir, file_name), 'a') as file:
            file.write(f'{server_round}, {self.cid}, {acc}, {loss}\n')
