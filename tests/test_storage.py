from backend.models.schemas import NotebookCreate
from backend.services.storage import NotebookStore


def test_create_list_get_rename_delete_notebook(tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))

    created = store.create(NotebookCreate(user_id="u1", name="Test"))
    assert created.user_id == "u1"
    assert created.created_at
    assert created.updated_at

    notebooks = store.list("u1")
    assert len(notebooks) == 1
    assert notebooks[0].notebook_id == created.notebook_id

    fetched = store.get("u1", created.notebook_id)
    assert fetched is not None
    assert fetched.name == "Test"

    renamed = store.rename("u1", created.notebook_id, "Renamed")
    assert renamed is not None
    assert renamed.name == "Renamed"

    assert store.get("u2", created.notebook_id) is None

    deleted = store.delete("u1", created.notebook_id)
    assert deleted is True
    assert store.get("u1", created.notebook_id) is None
    assert store.list("u1") == []


def test_invalid_user_id_rejected(tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))

    try:
        store.list("../bad")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
