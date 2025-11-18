git pull origin main
sudo docker build -t model -f dockerfile_model .
sudo docker run --name tf --rm -it model python3 model.py
