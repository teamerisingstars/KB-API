from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, field_validator


@dataclass
class Section:
    heading: str
    body: str
    source_file: str
    tokens: list[str]


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class SectionResult(BaseModel):
    answer: str
    section: str
    source: str
    confidence: float


class AskResponse(BaseModel):
    answer: Optional[str] = None
    section: Optional[str] = None
    source: Optional[str] = None
    confidence: float
    alternatives: list[SectionResult]
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    indexed_sections: int
    indexed_files: int
    last_indexed: Optional[str] = None


class SectionInfo(BaseModel):
    heading: str
    source: str


class SectionsResponse(BaseModel):
    sections: list[SectionInfo]


class ReloadResponse(BaseModel):
    status: str
    message: str
