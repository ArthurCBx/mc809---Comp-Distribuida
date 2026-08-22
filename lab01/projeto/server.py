import grpc
from concurrent import futures
import time
from sklearn.linear_model import LogisticRegression

import fit_train_pb2 as pb2
import fit_train_pb2_grpc as pb2_grpc

model = None

class FitTrainServicer(pb2_grpc.FitTrainServicer):
    def Train(self, request, context):
        X_train = [list(sample.features) for sample in request.samples]
        y_train = [sample.label for sample in request.samples]

        global model
        model = LogisticRegression()

        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        accuracy = model.score(X_train, y_train)

        return pb2.TrainOutput(accuracy=accuracy, time=elapsed_time)

class PredictServicer(pb2_grpc.PredictServicer):
    def Predict(self, request, context):
        global model
        if model is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Model not trained yet. Call Train first.")
            return pb2.PredictOutput()

        X_test = [list(sample.features) for sample in request.samples]
        y_pred = model.predict(X_test)

        return pb2.PredictOutput(predictions=[int(p) for p in y_pred])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_FitTrainServicer_to_server(FitTrainServicer(), server)
    pb2_grpc.add_PredictServicer_to_server(PredictServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
