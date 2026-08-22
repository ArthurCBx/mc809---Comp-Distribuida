FROM python:3.12-slim

# Ferramentas de sistema uteis dentro do devcontainer:
#  - git: integracao de source control do VS Code
#  - curl / ca-certificates: rede e TLS
#  - build-essential: caso alguma dependencia precise compilar
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv: copiado do binario oficial (nao usa pip para instalar o proprio uv).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# A venv gerenciada pelo uv fica FORA do bind mount do projeto (/opt/venv),
# para nao colidir com a .venv do host, que e do Windows e nao funciona no
# Linux do container. `uv sync` respeita UV_PROJECT_ENVIRONMENT e instala ali;
# colocar /opt/venv/bin no PATH faz o `python` do terminal ja usar essa venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_LINK_MODE=copy

# Porta padrao do servidor gRPC (servidor e cliente rodam neste container).
EXPOSE 50051

WORKDIR "/mc809 - Comp Distribuida"
