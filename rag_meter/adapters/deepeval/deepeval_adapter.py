import asyncio
import time
from typing import List

from fastapi import HTTPException, status
from loguru import logger

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)

from rag_meter.adapters.deepeval.models import (
    DeepEvalGeneration,
    DeepEvalResponse,
    DeepEvalRetrieval,
)
from rag_meter.adapters.models import RAGMeterRequest


class DeepEvalRAGEvaluationService:
    """
    A class responsible for evaluating RAG system responses using DeepEval metrics.

    This class takes in a request containing question-answer pairs, retrieval results,
    and generates DeepEval scores using various metrics. The evaluation is done
    concurrently for improved performance.
    """

    def __init__(self, request: RAGMeterRequest):
        """
        Initializes the evaluator with the provided RAGMeterRequest.

        Args:
            request (RAGMeterRequest): The request containing QA pairs, ground truths,
                                        LLM answers, and retrieved contexts.
        """
        self.request = request
        self.evaluator_model = request.evaluator_model

    async def deepeval_evaluation(self) -> List[dict]:
        """
        Evaluates the RAG responses using DeepEval metrics concurrently.

        This function evaluates the following metrics:
        - Answer Relevancy
        - Faithfulness
        - Contextual Recall
        - Contextual Precision
        - Contextual Relevancy

        Returns:
            List[dict]: A list of evaluation results with scores for each metric.
        """
        test_cases = self._create_test_cases()
        results = []

        for i, test_case in enumerate(test_cases):
            try:
                # Create fresh metric instances for each test case
                metrics = self._initialize_metrics()

                # Run metrics concurrently for this test case
                await asyncio.gather(
                    *[metric.a_measure(test_case) for metric in metrics]
                )

                # Store results per QA pair
                result = self._process_results(metrics)
                results.append(result)

            except Exception as e:
                logger.error(f"Error in metric calculation for QA pair {i}: {str(e)}")

        return results

    def _create_test_cases(self) -> List[LLMTestCase]:
        """
        Creates a list of test cases from the provided request.

        Returns:
            List[LLMTestCase]: List of test cases for each question-answer pair.
        """
        return [
            LLMTestCase(
                input=question,
                expected_output=answer,
                actual_output=llm_ans,
                retrieval_context=chunks,
            )
            for question, answer, llm_ans, chunks in zip(
                self.request.questions,
                self.request.ground_truth,
                self.request.llm_answers,
                self.request.retrieved_chunks,
            )
        ]

    def _initialize_metrics(self):
        """
        Initializes the metrics required for evaluation.

        Returns:
            List: A list of initialized metric instances.
        """
        return [
            AnswerRelevancyMetric(model=self.evaluator_model, async_mode=True),
            FaithfulnessMetric(model=self.evaluator_model, async_mode=True),
            ContextualRecallMetric(model=self.evaluator_model, async_mode=True),
            ContextualPrecisionMetric(model=self.evaluator_model, async_mode=True),
            ContextualRelevancyMetric(model=self.evaluator_model, async_mode=True),
        ]

    def _process_results(self, metrics) -> dict:
        """
        Processes the evaluation results from the metrics.

        Args:
            metrics (List): List of metric instances.

        Returns:
            dict: Processed results with rounded evaluation scores.
        """
        return {
            "answer_relevancy": round(float(metrics[0].score), 3),
            "faithfulness": round(float(metrics[1].score), 3),
            "contextual_recall": round(float(metrics[2].score), 3),
            "contextual_precision": round(float(metrics[3].score), 3),
            "contextual_relevancy": round(float(metrics[4].score), 3),
        }

    async def evaluate(self) -> List[DeepEvalResponse]:
        """
        Evaluate RAG system using DeepEval metrics and return formatted results.

        This method performs the following steps:
        - Evaluates the QA pairs using the DeepEval metrics.
        - Formats the results into a list of `DeepEvalResponse`.

        Returns:
            List[DeepEvalResponse]: List of formatted evaluation results.
        """
        try:
            logger.info("Starting evaluation with DeepEval...")
            start_time = time.time()

            results = []
            scores = await self.deepeval_evaluation()

            for question, answer, contexts, response, score_record in zip(
                self.request.questions,
                self.request.ground_truth,
                self.request.retrieved_chunks,
                self.request.llm_answers,
                scores,
            ):
                results.append(
                    DeepEvalResponse(
                        question=question,
                        expected_answer=answer,
                        generated_answer=response,
                        retrieved_contexts=contexts,
                        generation=DeepEvalGeneration(
                            faithfulness=score_record["faithfulness"],
                            answer_precision=score_record["answer_relevancy"],
                        ),
                        retrieval=DeepEvalRetrieval(
                            context_relevancy=score_record["contextual_relevancy"],
                            context_precision=score_record["contextual_precision"],
                            context_recall=score_record["contextual_recall"],
                        ),
                    )
                )

            logger.info(
                f"DeepEval Evaluation completed in: {(time.time() - start_time):.2f} seconds"
            )
            return results

        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            ) from e
