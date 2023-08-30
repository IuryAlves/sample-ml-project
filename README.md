## Installing

`pip install -r requirements.txt`

## Building

```sh
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 src/model_to_bento.py
bentoml build
bentoml containerize \
--enable-features grpc,grpc-reflection \
        -t test_model test_model
```

## Running

```sh
docker run -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python --rm -p 3000:3000 -p 3001:3001 test_model serve-grpc
```
