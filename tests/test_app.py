import pytest
import math
from app.app import app
from pathlib import Path
import os
from dotenv import load_dotenv
from app.model import load_model, predict
import xgboost


load_dotenv()


def test_model_path():
    model_path = os.getenv("MODEL_PATH")
    assert Path(model_path).exists()


def test_load_model():
    model_path = os.getenv("MODEL_PATH")
    model = load_model(model_path)
    assert isinstance(model, xgboost.sklearn.XGBRegressor)


def test_prediction_model():
    model_path = os.getenv("MODEL_PATH")
    model = load_model(model_path)
    data = {
        "single": 10,
        "married_no_kids": 5,
        "not_married_no_kids": 2,
        "married_with_kids": 8,
        "not_married_with_kids": 3,
        "single_parent": 1,
        "other": 0,
        "total": 29,
    }
    prediction = model.predict([list(data.values())])[0]
    print(f"{prediction=}")
    assert math.isclose(prediction, 1219730.9, rel_tol=1e-2)


def test_predict():
    model_path = os.getenv("MODEL_PATH")
    model = load_model(model_path)
    data = {
        "single": 10,
        "married_no_kids": 5,
        "not_married_no_kids": 2,
        "married_with_kids": 8,
        "not_married_with_kids": 3,
        "single_parent": 1,
        "other": 0,
        "total": 29,
    }
    prediction = predict(model, data)
    assert math.isclose(prediction, 1219730.9, rel_tol=1e-2)


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_get_woz_value(client):
    response = client.get(
        "/api/get_woz_value",
        query_string={
            "single": 10,
            "married_no_kids": 5,
            "not_married_no_kids": 2,
            "married_with_kids": 8,
            "not_married_with_kids": 3,
            "single_parent": 1,
            "other": 0,
            "total": 29,
        },
    )
    assert response.status_code == 200
    assert "woz_value" in response.json


def test_get_woz_prediction(client):
    response = client.get(
        "/api/get_woz_value",
        query_string={
            "single": 10,
            "married_no_kids": 5,
            "not_married_no_kids": 2,
            "married_with_kids": 8,
            "not_married_with_kids": 3,
            "single_parent": 1,
            "other": 0,
            "total": 29,
        },
    )
    assert response.status_code == 200
    assert "woz_value" in response.json
    prediction = response.json["woz_value"]
    assert math.isclose(float(prediction), 1219730.9, rel_tol=1e-2)


def test_missing_parameter(client):
    response = client.get(
        "/api/get_woz_value",
        query_string={
            "single": 10,
            "married_no_kids": 5,
            "not_married_no_kids": 2,
            "married_with_kids": 8,
            "not_married_with_kids": 3,
            "single_parent": 1,
            "other": 0,
        },
    )  # 'total' is missing
    assert response.status_code == 500
    assert "error" in response.json


def test_invalid_parameter_format(client):
    response = client.get(
        "/api/get_woz_value",
        query_string={
            "single": "wrong_format",
            "married_no_kids": 5,
            "not_married_no_kids": 2,
            "married_with_kids": 8,
            "not_married_with_kids": 3,
            "single_parent": 1,
            "other": 0,
            "total": 29,
        },
    )
    assert response.status_code == 500
    assert "error" in response.json
