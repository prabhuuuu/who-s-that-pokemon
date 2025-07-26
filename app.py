# app.py

from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load models
with open('model/pokemon_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open("model/name_predictor.pkl", "rb") as f:
    name_clf = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(feat)) for feat in [
            'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Generation'
        ]]

        prediction = model.predict([features])[0]
        result = "Legendary Pokémon" if prediction == 1 else "Not Legendary"

        guessed_name = name_clf.predict([features])[0]

        full_result = f"{result} - Guessed Name: {guessed_name}"

    except Exception as e:
        full_result = f"Error during prediction: {e}"

    return render_template('index.html', prediction=full_result)

if __name__ == '__main__':
    app.run(debug=True)
