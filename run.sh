git pull origin main

sudo docker build -t model -f dockerfile_model .
sudo docker run --name model --rm -it model python3 model.py

sudo docker build -t db -f dockerfile_db .
sudo docker run -p 8000:8000 db
