from typing import List
from pydantic import BaseModel


class DeepEvalRetrieval(BaseModel):
    context_relevancy: float
    context_precision: float
    context_recall: float


class DeepEvalGeneration(BaseModel):
    answer_precision: float
    faithfulness: float


class DeepEvalResponse(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: List[str]
    generation: DeepEvalGeneration
    retrieval: DeepEvalRetrieval
