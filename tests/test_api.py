import pytest
from fastapi.testclient import TestClient
from api.main import app, HTML_FILE


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "threshold" in data
    assert 0.0 <= data["threshold"] <= 1.0


def test_html_ui_serving(client):
    assert HTML_FILE.exists()
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn Decision System" in response.text

    ui_response = client.get("/ui")
    assert ui_response.status_code == 200
    assert "Customer Churn Decision System" in ui_response.text


def test_predict_single_high_risk(client):
    payload = {
        "Tenure": 1.0,
        "WarehouseToHome": 25.0,
        "NumberOfDeviceRegistered": 5,
        "SatisfactionScore": 1,
        "NumberOfAddress": 5,
        "DaySinceLastOrder": 1.0,
        "CashbackAmount": 110.0,
        "PreferedOrderCat": "Mobile Phone",
        "MaritalStatus": "Single",
        "Complain": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "threshold_used" in data
    assert "recommended_action" in data
    assert data["churn_prediction"] in [0, 1]


def test_predict_batch(client):
    payload = [
        {
            "Tenure": 2.0,
            "WarehouseToHome": 15.0,
            "NumberOfDeviceRegistered": 4,
            "SatisfactionScore": 3,
            "NumberOfAddress": 3,
            "DaySinceLastOrder": 1.0,
            "CashbackAmount": 145.0,
            "PreferedOrderCat": "Mobile Phone",
            "MaritalStatus": "Single",
            "Complain": 1,
        },
        {
            "Tenure": 25.0,
            "WarehouseToHome": 5.0,
            "NumberOfDeviceRegistered": 2,
            "SatisfactionScore": 5,
            "NumberOfAddress": 1,
            "DaySinceLastOrder": 10.0,
            "CashbackAmount": 220.0,
            "PreferedOrderCat": "Laptop & Accessory",
            "MaritalStatus": "Married",
            "Complain": 0,
        },
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert "churn_probability" in item
        assert "churn_prediction" in item
