import flask
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


graph = tf.get_default_graph()
app = Flask(__name__)
 

@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    data = [request.form['message']]
    sequences = tokenizer.texts_to_sequences(data)
    data_padded = pad_sequences(sequences,maxlen = 28, padding = 'post', truncating='post')

    global graph
    with graph.as_default():
        model = load_model('./static/model/model.h5')
        predictions = model.predict(data_padded)
   

    return render_template('index.html', pred=predictions)

def main():
    """Run the app."""
    app.run(debug=True)  # nosec

if __name__ == '__main__':
    main()