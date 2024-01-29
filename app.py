from flask import Flask, request, jsonify
from keras.models import load_model
from openai import OpenAI
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS

load_dotenv()  # Load environment variables from .env file

import os

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

app = Flask(__name__)
CORS(app, resources={r"/generate_content": {"origins": "*"}})  # Allow all origins for /generate_content route

# Load the Keras model, the scaler, and the label encoder
model = load_model('my_multiclass_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed_data = preprocess_data(data)

    # Make a prediction
    prediction = model.predict(processed_data)

    # Postprocess the prediction to get the class label
    classified_label = postprocess_prediction(prediction)

    # Return the prediction as a JSON response
    return jsonify({'prediction': classified_label})

def preprocess_data(data):
     # List of features to be dropped (identified from your training notebook)
    dropped_features = ['Date', 'Label', 'Label2', 'Label3', 'RoCL8', 'OBV', 'LaggedS15EMA', 'StdDevS15','MACDSigLine','MACDLine','CurrentPrice', 'RoCS4', 'ProfitLoss','ATR',]
   
    # Remove the dropped features from the data
    for feature in dropped_features:
        data.pop(feature, None)

    # Convert the remaining data to a NumPy array
    features = np.array([list(data.values())])

    # Apply scaling using the pre-fitted scaler
    scaled_features = scaler.transform(features)

    # Reshape the data to match the input shape expected by the model
    reshaped_features = scaled_features.reshape((1, 1, -1))
    return reshaped_features

def postprocess_prediction(prediction):
    # Get the index of the highest probability
    class_index = np.argmax(prediction, axis=-1)

    # Convert index to original class label using the label encoder
    original_class = label_encoder.inverse_transform(class_index)

    # Convert NumPy int64 to Python native int for JSON serialization
    return int(original_class[0])


@app.route('/generate_content', methods=['POST'])
def generate_content():
    try:
        # Get the user's input from the request
        user_input = request.json.get('user_input')
        # Use OpenAI's GPT-3 to generate content based on the user's input 
        print(user_input)
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": user_input}
        ]
        )
        # Extract the generated content from the response
        generated_content = completion.choices[0].message.content
        # Return the generated content as a JSON response
        return jsonify({'generated_content': generated_content})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
