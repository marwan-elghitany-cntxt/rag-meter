from typing import List
from pydantic import BaseModel


class TruLensEvalGeneration(BaseModel):
    honest_score: float
    harmless_score: float
    helpful_score: float


class TruLensEvalRetrieval(BaseModel):
    pass


class TruLensEvaluationResponse(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: List[str]
    generation: TruLensEvalGeneration
    retrieval: TruLensEvalRetrieval
