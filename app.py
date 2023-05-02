import openai
from flask import Flask, render_template, jsonify, request, Markup
from model import predict_image
import utils
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to Plant Disease Detection API'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    print(message)
    response = openai.Completion.create(
        engine='davinci', prompt=message, max_tokens=1024, n=1, stop=None, temperature=0.5
    )

    return jsonify({'message': response.choices[0].text.strip()})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print(request.files['file'])
        try:
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            res = Markup(utils.disease_dic[prediction])
            return res
        except:
            pass
    return jsonify({'message': 'Failed to predict'})


if __name__ == "__main__":
    app.run(debug=True,port=5000, host='0.0.0.0')
