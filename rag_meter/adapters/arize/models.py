from typing import List
from pydantic import BaseModel


class ArizeEvalMetric(BaseModel):
    label: str
    score: float
    explanation: str


class ArizeEvalGeneration(BaseModel):
    hallucination_eval: ArizeEvalMetric
    qa_correctness_eval: ArizeEvalMetric
    toxicity_eval: ArizeEvalMetric


class ArizeEvalRetrieval(BaseModel):
    pass


class ArizeEvaluationResponse(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: List[str]
    generation: ArizeEvalGeneration
    retrieval: ArizeEvalRetrieval
