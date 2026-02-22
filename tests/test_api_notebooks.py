from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_create_and_get_notebook():
    payload = {"user_id": "u1", "name": "Notebook 1"}
    resp = client.post("/api/notebooks/", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u1"
    assert data["name"] == "Notebook 1"
    assert "notebook_id" in data

    notebook_id = data["notebook_id"]
    resp_get = client.get(f"/api/notebooks/{notebook_id}")
    assert resp_get.status_code == 200
    data_get = resp_get.json()
    assert data_get["notebook_id"] == notebook_id


def test_get_missing_notebook():
    resp = client.get("/api/notebooks/missing")
    assert resp.status_code == 404
