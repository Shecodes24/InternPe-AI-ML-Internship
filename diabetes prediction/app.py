from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    output = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)
