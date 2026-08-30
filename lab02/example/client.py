import grpc
import ping_pong_pb2 as pb2
import ping_pong_pb2_grpc as pb2_grpc
import ray
import random

ray.init()

@ray.remote
class PingPongActor:
    def __init__(self, message, client_id):
        self.message = message
        self.client_id = client_id
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = pb2_grpc.PingPongStub(self.channel)

    def send_message(self):
        rnd = random.randint(0, 100)

        #altera ID com probabilidade de 50%
        if rnd < 50:
            self.client_id += 100

        message = pb2.Ping(mensagem=f'{self.message} - {self.client_id}')
        response = self.stub.GetServerResponse(message)
        return response.mensagem  

if __name__ == '__main__':
    clients_list = []

    for i in range(10):
        clients_list.append(PingPongActor.remote(f'Ping!', i))

    # Send messages using the actor    
    results = ray.get([client.send_message.remote() for client in clients_list])
    print(results)