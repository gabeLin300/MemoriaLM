from pydantic import BaseModel


class NotebookCreate(BaseModel):
    user_id: str
    name: str


class NotebookRename(BaseModel):
    user_id: str
    name: str


class NotebookOut(BaseModel):
    notebook_id: str
    user_id: str
    name: str
    created_at: str
    updated_at: str
