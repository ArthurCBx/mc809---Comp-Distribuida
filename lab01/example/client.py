import grpc
import ping_pong_pb2 as pb2
import ping_pong_pb2_grpc as pb2_grpc
import time

class ExemploGRPC(object):

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub        = pb2_grpc.PingPongStub(self.channel)

    def get_ping_response(self, message):
        message = pb2.Ping(mensagem=message)

        return self.stub.GetServerResponse(message)

if __name__ == '__main__':
    client   = ExemploGRPC()
    mensagem = 'Ping!'
    while True:
        tempo = time.time()
        print(f'Cliente -> {mensagem} {tempo}')
        resposta = client.get_ping_response(mensagem)
        print(f'Servidor -> {resposta.mensagem} {resposta.tempo}')
        print(f'Duração Ping -> Pong: {time.time() - tempo}')
        print('--------------------------------')
        time.sleep(1)