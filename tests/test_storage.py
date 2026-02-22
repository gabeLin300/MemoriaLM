from backend.services.storage import NotebookStore
from backend.models.schemas import NotebookCreate


def test_create_get_delete_notebook(tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))
    created = store.create(NotebookCreate(user_id="u1", name="Test"))

    fetched = store.get(created.notebook_id)
    assert fetched is not None
    assert fetched.notebook_id == created.notebook_id
    assert fetched.user_id == "u1"
    assert fetched.name == "Test"

    deleted = store.delete(created.notebook_id)
    assert deleted is True
    assert store.get(created.notebook_id) is None
