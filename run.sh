git pull origin main

sudo docker build -t model -f dockerfile_model .
sudo docker run --name model --rm -it model python3 model.py

sudo docker build -t db -f dockerfile_db .
sudo docker image ls
sudo docker run -p 8000:8000 db python db.py
sudo docker ps -a 

sudo docker pull ghcr.io/mlflow/mlflow
sudo docker run -p 8000:8001 -it ghcr.io/mlflow/mlflow
# 158.160.61.241:8001