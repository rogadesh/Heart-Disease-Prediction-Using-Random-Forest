import pickle
import numpy as np
import pandas as pd
import os

def load_model():
    with open("models/model.pkl", "rb") as f:
        scaler, model = pickle.load(f)
    return scaler, model

def predict_heart_disease_for_csv(csv_file):
    scaler, model = load_model()
    data = pd.read_csv(csv_file)
    input_scaled = scaler.transform(data.drop('target', axis=1, errors='ignore'))
    predictions = model.predict(input_scaled)

    # Replace 0/1 with meaningful text
    data['prediction'] = ["The person has heart disease" if pred == 1 else "The person does not have heart disease" for pred in predictions]

    if not os.path.exists('results'):
        os.makedirs('results')

    output_file = 'results/predictions.csv'
    data.to_csv(output_file, index=False)
    return output_file
