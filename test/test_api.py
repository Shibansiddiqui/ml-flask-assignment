import pytest

from app import app


@pytest.fixture
def client():

    app.config["TESTING"] = True

    with app.test_client() as client:

        yield client


# ------------------------------------------------------------
# Test English input
# ------------------------------------------------------------

def test_english_prediction(client):

    response = client.post(
        "/predict",
        json={
            "text": "I am feeling sick today."
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "category" in data
    assert "confidence" in data
    assert "response" in data
    assert "language" in data

    assert data["language"] == "English"


# ------------------------------------------------------------
# Test Hinglish input
# ------------------------------------------------------------

def test_hinglish_prediction(client):

    response = client.post(
        "/predict",
        json={
            "text": "Mujhe aaj bahut thakan ho rahi hai."
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "category" in data
    assert "confidence" in data
    assert "response" in data
    assert "language" in data

    assert data["language"] == "Hinglish"


# ------------------------------------------------------------
# Test empty input
# ------------------------------------------------------------

def test_empty_input(client):

    response = client.post(
        "/predict",
        json={
            "text": ""
        }
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


# ------------------------------------------------------------
# Test missing text
# ------------------------------------------------------------

def test_missing_input(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


# ------------------------------------------------------------
# Test invalid input
# ------------------------------------------------------------

def test_invalid_input(client):

    response = client.post(
        "/predict",
        json={
            "text": 12345
        }
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


# ------------------------------------------------------------
# Test response structure
# ------------------------------------------------------------

def test_response_structure(client):

    response = client.post(
        "/predict",
        json={
            "text": "Mujhe office late pahuchna hai"
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    expected_keys = [
        "category",
        "confidence",
        "response",
        "language"
    ]

    for key in expected_keys:

        assert key in data


# ------------------------------------------------------------
# Test confidence range
# ------------------------------------------------------------

def test_confidence_range(client):

    response = client.post(
        "/predict",
        json={
            "text": "I have a headache today."
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    confidence = data["confidence"]

    assert confidence >= 0
    assert confidence <= 1