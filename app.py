from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
heuristic_model = joblib.load('heuristic.joblib')
knn_model = joblib.load("knn.joblib")
rf_model = joblib.load('rf.joblib')
nn_model = tf.keras.models.load_model('model_nn_with_dropout.h5')

# Defines the validation for input data
def valid_input(input:list)-> bool:
    """
    Parameters:
        input (list): A list of 54 numbers representing the features of the input data.

    Returns:
        True if the input data is valid, and False otherwise.
    """
    if len(input) != 54:
        return False
    for num in input:
        if not isinstance(num, (int,float)):
            return False
    for num in input[11:]:
        if num not in [1,0]:
            return False
    return True

# Define the decoder function that decodes integer prediction to class label
def decoder(prediction:int) -> str:
    """
    Decode integer prediction to class label.

    Args:
        prediction (int): The integer prediction.

    Returns:
        str: The class label.
    """
    return ["Spruce/Fir","Lodgepole Pine","Ponderosa Pine","Cottonwood/Willow",
            "Aspen","Douglas-fir","Douglas-fir"][prediction-1]

# Define the predict function that predicts the class label
@app.route('/predict', methods=['POST'])
def predict() -> dict:
    """
    Predict the class label.

    Args:
        features (list): A list of features used for prediction.
        model (str): A string indicating the name of the model to use for prediction.
        decode (boolean): A boolean indicating whether to decode the prediction from an integer to a string class label.
    
    Returns:
        dict: The predicted class label.
    """
    # Get the input data
    data = request.get_json(force=True)
    
    #Validate data
    if not request.json or not valid_input(request.json['features']):
        return jsonify({'error': 'Invalid input data'}), 400
    
    features = np.array(data['features']).reshape(1, -1)

    # Choose the model  
    model_name = data['model']
    if model_name == 'Heuristic':
        prediction = heuristic_model.predict(features[:,[0]])[0]
    elif model_name == 'K Nearest Neighbors':
        prediction = knn_model.predict(features)[0]
    elif model_name == 'Random Forest':
        prediction = rf_model.predict(features)[0]
    elif model_name == 'Neural Network':
        prediction = nn_model.predict(features).argmax() +1 # Add 1 since NN returns values from 0-6.
    else:
        return jsonify({'error': 'Wrong model name'}), 400

    # Return the prediction
    if data['decode'] == True:
        return jsonify({"prediction": str(decoder(prediction))}) , 200
    return jsonify({'prediction': int(prediction)}) , 200

if __name__ == '__main__':
    app.run(debug=True)
