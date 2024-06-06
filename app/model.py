import xgboost as xgb

def load_model(model_path):
    """
    The function `load_model` loads an XGBoost regressor model from a specified file path.
    
    :param model_path: The `model_path` parameter in the `load_model` function is the file path where
    the XGBoost model is saved. This function loads the XGBoost model from the specified file path using
    the `load_model` method of the `XGBRegressor` class
    :return: The function `load_model` returns an XGBoost regressor model loaded from the specified
    model path.
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def predict(model, data):
    """
    The `predict` function takes a model and data as input, predicts the outcome using the model, and
    returns the prediction.
    
    :param model: A machine learning model that has been trained on a dataset and is capable of making
    predictions on new data
    :param data: It seems like you were about to provide some information about the `data` parameter,
    but it got cut off. Could you please provide more details or specific examples of what the `data`
    parameter might contain? This will help me assist you better with the `predict` function
    :return: The function `predict` returns the prediction made by the model for the input data
    provided.
    """
    prediction = model.predict([list(data.values())])
    return prediction[0]