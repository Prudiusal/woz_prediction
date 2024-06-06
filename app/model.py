import xgboost as xgb


def load_model(model_path):
    """
    :param model_path: path of the model
    :return: returns an XGBoost regressor model loaded from the specified
    model path.
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def predict(model, data):
    """

    :param model: A machine learning model that has been trained on a dataset
    :param data: request data in the form of a dictionary
    :return: The function `predict` returns the numerical prediction.
    """
    prediction = model.predict([list(data.values())])
    return prediction[0]
