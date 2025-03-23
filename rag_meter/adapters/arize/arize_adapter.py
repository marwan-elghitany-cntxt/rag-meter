import asyncio
import time
from fastapi import HTTPException, status
import pandas as pd
from typing import List
from loguru import logger
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    ToxicityEvaluator,
    OpenAIModel,
    run_evals,
)

from rag_meter.adapters.arize.models import (
    ArizeEvalGeneration,
    ArizeEvalMetric,
    ArizeEvalRetrieval,
    ArizeEvaluationResponse,
)
from rag_meter.adapters.models import RAGMeterRequest


class ArizeEvaluationService:
    def __init__(self, request: RAGMeterRequest, model_name: str = "gpt-4o-mini"):
        """Initialize the evaluation model and evaluators."""
        self.request = request
        self.eval_model = OpenAIModel(model=model_name)
        self.hallucination_evaluator = HallucinationEvaluator(self.eval_model)
        self.qa_correctness_evaluator = QAEvaluator(self.eval_model)
        self.toxicity_evaluator = ToxicityEvaluator(self.eval_model)

    async def evaluate(self) -> List[ArizeEvaluationResponse]:
        """Evaluate LLM responses using multiple metrics and return structured results."""
        try:
            start_time = time.time()
            logger.info("Evaluating with Arize in batch-async mode...")

            # Prepare batch DataFrame for evaluation
            df_batch = pd.DataFrame(
                {
                    "input": self.request.questions,
                    "reference": self.request.ground_truth,
                    "output": self.request.llm_answers,
                }
            )

            # Run evaluations asynchronously
            hallucination_df, qa_correctness_df, toxicity_df = await asyncio.to_thread(
                run_evals,
                dataframe=df_batch,
                evaluators=[
                    self.hallucination_evaluator,
                    self.qa_correctness_evaluator,
                    self.toxicity_evaluator,
                ],
            )

            # Construct structured responses
            results = [
                self._generate_response(
                    i,
                    question,
                    answer,
                    generated,
                    retrieved_chunk,
                    hallucination_df,
                    qa_correctness_df,
                    toxicity_df,
                )
                for i, (question, answer, generated, retrieved_chunk) in enumerate(
                    zip(
                        self.request.questions,
                        self.request.ground_truth,
                        self.request.llm_answers,
                        self.request.retrieved_chunks,
                    )
                )
            ]

            logger.info(
                f"Arize batch evaluation completed in {time.time() - start_time:.2f}s"
            )
            return results

        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            ) from e

    def _generate_response(
        self,
        index: int,
        question: str,
        answer: str,
        generated: str,
        retrieved_chunk: str,
        hallucination_df: pd.DataFrame,
        qa_correctness_df: pd.DataFrame,
        toxicity_df: pd.DataFrame,
    ) -> ArizeEvaluationResponse:
        """Generate a structured evaluation response."""
        return ArizeEvaluationResponse(
            question=question,
            expected_answer=answer,
            generated_answer=generated,
            retrieved_contexts=retrieved_chunk,
            generation=ArizeEvalGeneration(
                hallucination_eval=ArizeEvalMetric(
                    label=hallucination_df.iloc[index]["label"],
                    score=hallucination_df.iloc[index]["score"],
                    explanation=hallucination_df.iloc[index].get("explanation", ""),
                ),
                qa_correctness_eval=ArizeEvalMetric(
                    label=qa_correctness_df.iloc[index]["label"],
                    score=qa_correctness_df.iloc[index]["score"],
                    explanation=qa_correctness_df.iloc[index].get("explanation", ""),
                ),
                toxicity_eval=ArizeEvalMetric(
                    label=toxicity_df.iloc[index]["label"],
                    score=toxicity_df.iloc[index]["score"],
                    explanation=toxicity_df.iloc[index].get("explanation", ""),
                ),
            ),
            retrieval=ArizeEvalRetrieval(),
        )
