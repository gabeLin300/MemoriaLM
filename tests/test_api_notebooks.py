from fastapi.testclient import TestClient

from backend.app import app
from backend.api import notebooks as notebooks_api
from backend.services.storage import NotebookStore


client = TestClient(app)
AUTH_U1 = {"X-User-Id": "u1"}
AUTH_U2 = {"X-User-Id": "u2"}


def setup_function():
    notebooks_api.store = NotebookStore(base_dir="tests_tmp_data")


def test_create_list_get_rename_delete_notebook(tmp_path):
    notebooks_api.store = NotebookStore(base_dir=str(tmp_path))

    payload = {"user_id": "u1", "name": "Notebook 1"}
    resp = client.post("/api/notebooks/", json=payload, headers=AUTH_U1)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u1"
    assert data["name"] == "Notebook 1"
    assert "notebook_id" in data

    notebook_id = data["notebook_id"]

    resp_list = client.get("/api/notebooks/", params={"user_id": "u1"}, headers=AUTH_U1)
    assert resp_list.status_code == 200
    listed = resp_list.json()
    assert len(listed) == 1
    assert listed[0]["notebook_id"] == notebook_id

    resp_get = client.get(f"/api/notebooks/{notebook_id}", params={"user_id": "u1"}, headers=AUTH_U1)
    assert resp_get.status_code == 200
    data_get = resp_get.json()
    assert data_get["notebook_id"] == notebook_id

    resp_wrong_user = client.get(f"/api/notebooks/{notebook_id}", params={"user_id": "u2"}, headers=AUTH_U2)
    assert resp_wrong_user.status_code == 404

    resp_rename = client.patch(
        f"/api/notebooks/{notebook_id}",
        json={"user_id": "u1", "name": "Renamed"},
        headers=AUTH_U1,
    )
    assert resp_rename.status_code == 200
    assert resp_rename.json()["name"] == "Renamed"

    resp_delete = client.delete(f"/api/notebooks/{notebook_id}", params={"user_id": "u1"}, headers=AUTH_U1)
    assert resp_delete.status_code == 200
    assert resp_delete.json() == {"deleted": True}


def test_get_missing_notebook(tmp_path):
    notebooks_api.store = NotebookStore(base_dir=str(tmp_path))
    resp = client.get("/api/notebooks/missing", params={"user_id": "u1"}, headers=AUTH_U1)
    assert resp.status_code == 404


def test_rejects_mismatched_user_id_payload(tmp_path):
    notebooks_api.store = NotebookStore(base_dir=str(tmp_path))
    resp = client.post(
        "/api/notebooks/",
        json={"user_id": "u2", "name": "Notebook 1"},
        headers=AUTH_U1,
    )
    assert resp.status_code == 403
