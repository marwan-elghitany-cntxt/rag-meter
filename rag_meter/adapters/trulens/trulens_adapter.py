import asyncio
import time
from typing import Dict, List

from fastapi import HTTPException, status
from trulens.providers.openai import OpenAI
from trulens.core import Feedback
from loguru import logger

from rag_meter.adapters.trulens.models import (
    TruLensEvalGeneration,
    TruLensEvalRetrieval,
    TruLensEvaluationResponse,
)
from rag_meter.adapters.models import RAGMeterRequest

import asyncio
import time
from typing import List, Dict


class TruLensEvaluationService:
    """Service for evaluating RAG models using TruLens metrics (Honest, Harmless, Helpful)."""

    def __init__(self, request: RAGMeterRequest):

        self.request = request
        self.provider = OpenAI()
        self.honest_feedback = Feedback(self.provider.correctness).on_input_output()
        self.harmless_feedback = Feedback(self.provider.harmfulness).on_response()
        self.helpful_feedback = Feedback(self.provider.helpfulness).on_input_output()

    async def _evaluate_response(
        self, question: str, response: str, context: List[str]
    ) -> Dict[str, float]:
        """
        Evaluates a response based on TruLens feedback metrics.

        Args:
            question (str): The original question.
            response (str): The generated response.
            context (List[str]): List of retrieved context chunks.

        Returns:
            Dict[str, float]: Scores for honest, harmless, and helpful metrics.
        """
        try:
            combined_context = " ".join(context)
            input_text = f"Context: {combined_context}\nQuestion: {question}"

            honest_score, harmless_score, helpful_score = await asyncio.gather(
                asyncio.to_thread(self.honest_feedback, input_text, response),
                asyncio.to_thread(self.harmless_feedback, response),
                asyncio.to_thread(self.helpful_feedback, input_text, response),
            )

            logger.info(
                f"Evaluation Scores - Honest: {honest_score}, Harmless: {harmless_score}, Helpful: {helpful_score}"
            )
            return {
                "honest_score": float(honest_score),
                "harmless_score": float(harmless_score),
                "helpful_score": float(helpful_score),
            }

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return {"honest_score": 0.0, "harmless_score": 0.0, "helpful_score": 0.0}

    async def evaluate(self) -> List[TruLensEvaluationResponse]:
        """
        Evaluates multiple QA pairs using TruLens metrics.

        Args:
            request (RAGMeterRequest): Request containing QA pairs and their context.

        Returns:
            List[TruLensEvaluationResponse]: Evaluation results for each QA pair.
        """
        try:
            start_time = time.time()
            logger.info("Starting TruLens evaluation...")

            tasks = [
                self._evaluate_response(question, answer, chunks)
                for question, answer, chunks, _ in zip(
                    self.request.questions,
                    self.request.llm_answers,
                    self.request.retrieved_chunks,
                    self.request.ground_truth,
                )
            ]

            scores_list = await asyncio.gather(*tasks)

            results = [
                TruLensEvaluationResponse(
                    question=question,
                    expected_answer=answer,
                    generated_answer=response,
                    retrieved_contexts=chunks,
                    generation=TruLensEvalGeneration(
                        honest_score=scores["honest_score"],
                        harmless_score=scores["harmless_score"],
                        helpful_score=scores["helpful_score"],
                    ),
                    retrieval=TruLensEvalRetrieval(),
                )
                for (question, answer, chunks, response), scores in zip(
                    zip(
                        self.request.questions,
                        self.request.llm_answers,
                        self.request.retrieved_chunks,
                        self.request.ground_truth,
                    ),
                    scores_list,
                )
            ]

            logger.info(
                f"TruLens Evaluation completed in {time.time() - start_time:.2f} seconds"
            )
            return results

        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            ) from e
