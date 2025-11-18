docker build -t server -f dockerfile_web .
docker run -d --name server -p 8000:8000 server python server.py
docker ps
curl -X GET http://localhost:8000/?url=https://i.pinimg.com/736x/01/f0/d2/01f0d2329d42c41e9b5cf315665783fd.jpg
