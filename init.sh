# apt-get update
# sudo apt-get install git 
# git clone https://github.com/ESSch/ML.git
# cd ML
sudo apt-get install docker.io
sudo docker version
sudo docker pull huggingface/transformers-pytorch-gpu
sudo docker image ls
sudo docker tag huggingface/transformers-pytorch-gpu:latest tf
sudo docker image ls