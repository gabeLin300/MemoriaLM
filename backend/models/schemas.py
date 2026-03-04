from typing import List, Literal, Optional

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
    enabled: bool = True
    raw_path: Optional[str] = None
    extracted_path: str
    chunk_count: int
    char_count: int
    created_at: str


class SourceListOut(BaseModel):
    sources: List[SourceOut]


class SourceToggleRequest(BaseModel):
    user_id: str
    enabled: bool


class ChatRequest(BaseModel):
    user_id: str
    message: str
    top_k: int = 5
    retrieval_mode: Literal["topk", "rerank"] = "topk"


class CitationOut(BaseModel):
    source_name: str
    source_type: str
    location: str
    chunk_id: str


class ChatResponseOut(BaseModel):
    answer: str
    citations: List[CitationOut]
    used_chunks: int


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: str
    citations: Optional[List[CitationOut]] = None


class ChatHistoryOut(BaseModel):
    messages: List[ChatMessageOut]


class ArtifactGenerateRequest(BaseModel):
    user_id: str
    prompt: Optional[str] = None
    num_questions: int = 8


class ArtifactFileOut(BaseModel):
    name: str
    path: str
    created_at: str


class PodcastArtifactOut(BaseModel):
    transcript: Optional[ArtifactFileOut] = None
    audio: Optional[ArtifactFileOut] = None


class ArtifactListOut(BaseModel):
    reports: List[ArtifactFileOut]
    quizzes: List[ArtifactFileOut]
    flashcards: List[ArtifactFileOut]
    podcasts: List[PodcastArtifactOut]


class ArtifactGenerateOut(BaseModel):
    artifact_type: str
    message: str
    markdown_path: str
    audio_path: Optional[str] = None
    created_at: str
