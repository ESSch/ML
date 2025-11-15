from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_get():
    return "Hello, world!"

if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0')