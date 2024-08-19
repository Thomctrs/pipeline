# app/main.py
from flask import Flask, request, jsonify
from anomaly_retrieval_pipeline import Pipeline  #

app = Flask(__name__)
pipeline = Pipeline()

@app.route('/')
def index():
    return "Bienvenue sur l'application Flask de pipeline !"

@app.route('/api/process/models', methods=['GET'])
def get_models():
    # Code pour retourner la liste des modèles ou des données requises
    return jsonify({'models': ['model1', 'model2']})


@app.route('/api/process', methods=['POST'])
def process_request():
    data = request.json
    user_message = data.get('user_message')
    model_id = data.get('model_id')
    messages = data.get('messages')
    body = data.get('body') 

    response = pipeline.pipe(user_message, model_id, messages, body)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
