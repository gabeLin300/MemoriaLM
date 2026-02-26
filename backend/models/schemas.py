from typing import List, Optional

from pydantic import BaseModel, HttpUrl


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


class UrlIngestRequest(BaseModel):
    user_id: str
    url: HttpUrl
    source_name: Optional[str] = None


class SourceOut(BaseModel):
    source_id: str
    source_name: str
    source_type: str
    raw_path: Optional[str] = None
    extracted_path: str
    chunk_count: int
    char_count: int
    created_at: str


class SourceListOut(BaseModel):
    sources: List[SourceOut]
