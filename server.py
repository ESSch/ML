from flask import Flask, request, jsonify
# import model

app = Flask(__name__)

# for test: curl -X GET http://localhost:8000/
# response: Hello, world!
# for test: curl -X GET http://localhost:8000/?name=Evgeniy
# response: Hello, Evgeniy!
# for test: curl -X GET http://localhost:8000/?url=https://i.pinimg.com/736x/01/f0/d2/01f0d2329d42c41e9b5cf315665783fd.jpg
# response: 
@app.route('/', methods=['GET'])
def hello_get():
    name = request.args.get('name', 'world')
    result = {};
    url = request.args.get('url', '');
    if name != '':
        result["response"] = f"Hello, {name}!";
    if url != '':
        result["response"] = f"Image to text: {url}";
        # result["response"] = model.main(url);
    return jsonify(result);

if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0')