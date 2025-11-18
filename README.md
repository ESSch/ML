# Docker-compose
```
docker compose -f Docker-compose up
docker compose -f Docker-compose down
```
# Server
```
docker build -t server -f dockerfile_web .
docker run -it -p 8000:8000 server python server.py
curl http://localhost:8000/
curl http://localhost:8000/?name=Evgeniy
```
# DB
```
docker run -v ./chroma-data:/data -p 800:8001 chromadb/chroma
curl http://localhost:8001/
```
# CI / CD
```
bash push.sh
bash run.sh
```
# Jupyter
```
docker run --rm ubuntu:18.04 echo 1 
docker build -t cv -f dockerfile_my .
docker run -it -p 8789:8789 -v $(pwd):/playground cv
docker ps
docker exec -it 5ad6135ad647 jupyter server list
http://localhost:8789?token=3e11bcf2bf4238d2ebcb553ef61c9142b2ec23a50a6084a0
cat /etc/lsb-release
import torch
print(torch.__version__)
torch.cuda.is_available()
```
# CUDA
```
docker build -t cuda -f Dockerfile_cuda .
docker image ls
docker run -it cuda nvidia-smi
```