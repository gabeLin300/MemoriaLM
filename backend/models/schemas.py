from pydantic import BaseModel


class NotebookCreate(BaseModel):
    user_id: str
    name: str


class NotebookOut(BaseModel):
    notebook_id: str
    user_id: str
    name: str
