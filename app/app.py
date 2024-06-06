import os

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from .model import load_model, predict


load_dotenv()

app = Flask(__name__)

model_path = os.getenv('MODEL_PATH')

if not model_path:
    raise ValueError("MODEL_PATH environment variable is not set")

# Load the model
model = load_model(model_path)

@app.route('/api/get_woz_value', methods=['GET'])
def get_woz_value():
    """
    The function `get_woz_value` takes input parameters from a request, validates them, makes a
    prediction using a model, and returns the predicted woz_value.
    :return: The `get_woz_value` function is returning a JSON response. If there are no errors, it
    returns a JSON object with the predicted 'woz_value'. If there is a `ValueError`, it returns a JSON
    object with the error message and a status code of 400. If there is any other exception, it returns
    a JSON object with the error message 'An error occurred' and
    """
    required_params = [
        'single', 
        'married_no_kids',
        'not_married_no_kids', 
        'married_with_kids', 
        'not_married_with_kids',
        'single_parent', 
        'other', 
        'total'
    ]

    try:
        
        data = {param: int(request.args[param]) for param in required_params}
        # print(f'{data=}')
        # print(f'{predict=}')
        # print(f'{model=}')
        # prediction = 5
        

        prediction = predict(model, data)
        return jsonify({'woz_value': str(prediction)}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred {e}'}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=51000)
