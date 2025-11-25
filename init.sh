# YandexCloud creat VM Ubuntu 2CPU 2RAM 300Gb User:essch SSH:essch Name:ml2
ssh -l essch 158.160.61.241
git clone https://github.com/ESSch/ML.git
sudo apt-get update
sudo apt-get install docker.io
docker -v
cd ML
git pull origin main