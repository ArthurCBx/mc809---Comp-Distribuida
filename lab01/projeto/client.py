from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import grpc
import fit_train_pb2 as pb2
import fit_train_pb2_grpc as pb2_grpc

class IrisClient(object):

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.train_stub  = pb2_grpc.FitTrainStub(self.channel)
        self.predict_stub = pb2_grpc.PredictStub(self.channel)

        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def fit_train_model(self):
        samples = [
            pb2.Sample(features=list(features), label=int(label))
            for features, label in zip(self.X_train, self.y_train)
        ]
        train_data = pb2.DataInput(samples=samples)

        return self.train_stub.Train(train_data)

    def predict_request(self):
        samples = [
            pb2.Sample(features=list(features), label=0)
            for features in self.X_test
        ]
        predict_request_data = pb2.PredictInput(samples=samples)

        return self.predict_stub.Predict(predict_request_data)

    def calculate_accuracy(self, predictions):
        correct_predictions = sum(
            1 for pred, true in zip(predictions, self.y_test) if pred == true
        )
        return correct_predictions / len(self.y_test)

if __name__ == '__main__':
    client = IrisClient()

    response = client.fit_train_model()
    print(f"Accuracy: {response.accuracy:.4f}")
    print(f"Training time: {response.time:.4f}s")

    response = client.predict_request()
    print(f"Predictions: {list(response.predictions)}")

    print(f"Calculated test accuracy: {client.calculate_accuracy(response.predictions):.4f}")
