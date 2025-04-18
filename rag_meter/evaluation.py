import asyncio
from typing import Any, Dict
from loguru import logger
from rag_meter.adapters.models import RAGMeterRequest
from rag_meter.adapters.ragas.ragas_adapter import RAGASEvaluationService
from rag_meter.adapters.deepeval.deepeval_adapter import DeepEvalRAGEvaluationService
from rag_meter.adapters.arize.arize_adapter import ArizeEvaluationService
from rag_meter.adapters.trulens.trulens_adapter import TruLensEvaluationService
from rag_meter.aggregator import RAGEvaluationAggregator
from rag_meter.constants import ARIZE, DEEPEVAL, RAGAS, TRULENS

import asyncio
from typing import List, Dict, Any, Optional


class RAGMeter:
    """Manages multi-framework RAG evaluations and aggregates results."""

    def __init__(
        self,
        questions: List[str],
        ground_truth: List[str],
        llm_answers: List[str],
        retrieved_chunks: List[List[str]],
        frameworks: Optional[List[str]] = None,
    ):
        """
        Initializes the RAGMeter with evaluation parameters.

        :param questions: List of questions asked.
        :param ground_truth: List of corresponding ground-truth answers.
        :param llm_answers: List of answers generated by the LLM.
        :param retrieved_chunks: Context retrieved for each question.
        :param frameworks: List of RAG evaluation frameworks to use (optional).
        """
        self.request = RAGMeterRequest(
            questions=questions,
            ground_truth=ground_truth,
            llm_answers=llm_answers,
            retrieved_chunks=retrieved_chunks,
            frameworks=frameworks or ["ragas", "arize", "deepeval", "trulens"],
        )

        self.rag_evaluators = {
            RAGAS: RAGASEvaluationService,
            DEEPEVAL: DeepEvalRAGEvaluationService,
            TRULENS: TruLensEvaluationService,
            ARIZE: ArizeEvaluationService,
        }

    async def _run_evaluations(self, frameworks: list) -> Dict[str, Any]:
        """Executes selected framework evaluations asynchronously."""
        tasks = {
            fw: self.rag_evaluators[fw](self.request).evaluate() for fw in frameworks
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Handle potential errors
        response_data = {}
        for fw, result in zip(frameworks, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {fw} evaluation: {result}")
                response_data[fw] = {"error": str(result)}
            else:
                response_data[fw] = result
        return response_data

    async def multi_evaluation(self) -> Dict[str, Any]:
        """Runs evaluations across multiple frameworks and aggregates results."""
        valid_frameworks = [
            fw for fw in self.request.frameworks if fw in self.rag_evaluators
        ]
        if not valid_frameworks:
            logger.warning("No valid frameworks provided for evaluation.")
            return {"error": "No valid frameworks selected."}

        logger.info(f"Running evaluations for: {valid_frameworks}")
        evaluation_results = await self._run_evaluations(valid_frameworks)

        logger.info("Aggregating evaluation results...")
        aggregated_results = RAGEvaluationAggregator(
            evaluation_results
        ).process_results()
        logger.info("Aggregation complete.")

        return aggregated_results
