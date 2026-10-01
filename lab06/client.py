import flwr as fl
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions, plot_comparison_label_distribution


class Cliente(fl.client.NumPyClient):
    def __init__(self, cid, niid, num_clients, dirichlet_alpha):
        self.cid             = int(cid)
        self.niid            = niid
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.model                                           = self.create_model(self.x_train.shape)

    def get_parameters(self, config):
        return self.model.get_weights()

    def load_data(self):
        if self.niid:
            partitioner_train = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=0,
                                    self_balancing=False)
            partitioner_test = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=0,
                                    self_balancing=False)
        else:
            partitioner_train =  IidPartitioner(num_partitions=self.num_clients)
            partitioner_test  = IidPartitioner(num_partitions=self.num_clients)

        fds               = FederatedDataset(dataset='mnist', partitioners={"train": partitioner_train})
        train             = fds.load_partition(self.cid).with_format("numpy")

        fds_eval          = FederatedDataset(dataset='mnist', partitioners={"test": partitioner_test})
        test              = fds_eval.load_partition(self.cid).with_format("numpy")

        return train['image']/255.0, train['label'], test['image']/255.0, test['label']

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
        self.model.set_weights(parameters)

        history = self.model.fit(self.x_train, self.y_train, epochs=1)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()

        fit_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss,
        }

        self.log_client('train.csv', config['server_round'], acc, loss)
        return trained_parameters, len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        self.log_client('evaluate.csv', config['server_round'], acc, loss)
        return loss, len(self.x_test), eval_msg

    def log_client(self, file_name, server_round, acc, loss):
        with open(file_name, 'a') as file:
            file.write(f'{server_round}, {self.cid}, {acc}, {loss}\n')
