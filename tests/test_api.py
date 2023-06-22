import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.models.request import Request


def test_endpoint():
    client = TestClient(app)

    prompt = "A painting of a dog"
    # paylod without weights
    payload = Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        weights=None,)
    
    response = client.post("/get_image", json=payload.dict())

    assert response.status_code == 200
    assert response.json()["request_id"] == payload.request_id
    assert response.json()["weights"] is not None

    # payload with weights
    payload2 = Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        weights=response.json()["weights"],)
    
    response2 = client.post("/get_image", json=payload2.dict())

    assert response2.status_code == 200
    assert response2.json()["request_id"] == payload2.request_id

    # weight should be different
    assert response2.json()["weights"] != response.json()["weights"]