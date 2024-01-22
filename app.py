from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
