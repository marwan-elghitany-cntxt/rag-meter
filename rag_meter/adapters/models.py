from typing import List, Optional
from pydantic import BaseModel


class RAGMeterRequest(BaseModel):
    llm_answers: List[str]
    questions: List[str]
    ground_truth: List[str]
    retrieved_chunks: List[List[str]]
    frameworks: List[str]
    evaluator_model: Optional[str] = "gpt-4o-mini"
