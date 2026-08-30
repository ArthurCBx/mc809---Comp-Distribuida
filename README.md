## Executável para compilar .proto
```bash
python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. <filename>.proto
```
> Must be executed from the folder where the `.proto` is stored

