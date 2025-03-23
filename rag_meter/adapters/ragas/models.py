from typing import List, Optional
from pydantic import BaseModel, Field


class RAGASEvalGeneration(BaseModel):
    answer_relevancy: float
    faithfulness: float


class RAGASEvalRetrieval(BaseModel):
    context_recall: float
    context_precision: float
    answer_correctness: float
    semantic_similarity: float


class RAGASEvaluationResponse(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: List[str]
    generation: RAGASEvalGeneration
    retrieval: RAGASEvalRetrieval
