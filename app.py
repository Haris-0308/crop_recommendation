from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained RandomForest model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the mapping of labels to crop names
crop_names = {
    0: 'rice',
    1: 'maize',
    2: 'chickpea',
    3: 'kidneybeans',
    4: 'pigeonpeas',
    5: 'mothbeans',
    6: 'mungbean',
    7: 'blackgram',
    8: 'lentil',
    9: 'pomegranate',
    10: 'banana',
    11: 'mango',
    12: 'grapes',
    13: 'watermelon',
    14: 'muskmelon',
    15: 'apple',
    16: 'orange',
    17: 'papaya',
    18: 'coconut',
    19: 'cotton',
    20: 'jute',
    21: 'coffee'
}

# Check if the model was loaded correctly
print(f"Model loaded: {type(model)}")

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict the crop recommendation
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting data from the form
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare the input DataFrame
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    print("Input Data:", input_data)  # Debug line

    # Make a prediction using the loaded model
    prediction_array = model.predict(input_data)  # Get prediction probabilities
    print("Prediction Array:", prediction_array)  # Debug line

    # Use argmax to find the predicted crop index
    predicted_crop_index = np.argmax(prediction_array, axis=1)[0]  # Get index of the highest prediction
    predicted_crop = crop_names.get(predicted_crop_index, "Unknown crop")  # Map to crop name

    print("Predicted Crop Index:", predicted_crop_index)  # Debug line
    print("Predicted Crop:", predicted_crop)  # Debug line

    # Return the result
    return render_template('index.html', prediction_text=f'Recommended Crop: {predicted_crop}')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

