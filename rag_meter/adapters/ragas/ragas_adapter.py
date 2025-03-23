import time
from typing import List

from fastapi import HTTPException, status
from loguru import logger

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from rag_meter.adapters.ragas.models import (
    RAGASEvalGeneration,
    RAGASEvalRetrieval,
    RAGASEvaluationResponse,
)
from rag_meter.adapters.models import RAGMeterRequest

from dotenv import load_dotenv
load_dotenv()


class RAGASEvaluationService:
    """
    RAGASEvaluator class is responsible for evaluating the responses of the LLM
    against the provided ground truth using several evaluation metrics.

    Attributes:
        request (RAGMeterRequest): The request object containing questions, answers, and retrieved contexts.
        data_sample (dict): A dictionary containing user input, response, retrieved contexts, and reference data.
    """

    def __init__(self, request: RAGMeterRequest):
        """
        Initializes the evaluator with a RAGMeterRequest object and prepares
        the data for evaluation.

        Args:
            request (RAGMeterRequest): The request object containing the evaluation data.
        """
        self.request = request
        self.data_sample = {
            "user_input": request.questions,
            "response": request.llm_answers,
            "retrieved_contexts": request.retrieved_chunks,
            "reference": request.ground_truth,
        }

    async def evaluate(self) -> List[RAGASEvaluationResponse]:
        """
        Evaluates the responses based on the metrics provided and formats
        the results into RAGASEvaluationResponse objects.

        This method orchestrates the evaluation and returns the result.

        Returns:
            List[RAGASEvaluationResponse]: A list of formatted evaluation responses.
        """
        try:
            logger.info("Starting response evaluation...")

            # Measure evaluation time
            start_time = time.time()

            # Perform the evaluation
            scores_df = await self.ragas_evaluation()

            # Format the results into response objects
            results = self.format_results(scores_df)

            # Log the time taken for evaluation
            logger.info(
                f"Evaluation completed in {(time.time() - start_time):.2f} seconds"
            )
            return results
        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during evaluation.",
            ) from e

    async def ragas_evaluation(self) -> List[dict]:
        """
        Performs the actual evaluation of the LLM responses using the provided data sample and evaluation metrics.

        Returns:
            List[dict]: A list of dictionaries containing evaluation results.
        """
        # Prepare the Langchain LLM wrapper and metrics
        logger.info("RAGAS EVALUATION ...")
        llm = LangchainLLMWrapper(
            ChatOpenAI(model=self.request.evaluator_model, temperature=0.1)
        )

        evaluation_result = evaluate(
            llm=llm,
            dataset=Dataset.from_dict(self.data_sample),
            metrics=[
                answer_correctness,
                answer_relevancy,
                answer_similarity,
                context_precision,
                context_recall,
                faithfulness,
            ],
            show_progress=True,
        )

        # Convert the evaluation result to a pandas DataFrame and then to a dictionary
        df = evaluation_result.to_pandas()
        df = (
            df.rename(
                columns={
                    "user_input": "question",
                    "reference": "ground_truth",
                    "response": "llm_answer",
                    "answer_correctness": "answer_precision",
                    "answer_relevancy": "context_relevancy",
                }
            )
            .round(3)
            .to_dict("records")
        )
        return df

    def format_results(self, scores_df: List[dict]) -> List[RAGASEvaluationResponse]:
        """
        Formats the raw evaluation data into a structured list of RAGASEvaluationResponse objects.

        Args:
            scores_df (List[dict]): The raw evaluation data in dictionary format.

        Returns:
            List[RAGASEvaluationResponse]: A list of structured evaluation response objects.
        """
        logger.info("Formatting Results ...")
        results = []
        for record in scores_df:
            results.append(
                RAGASEvaluationResponse(
                    question=record["question"],
                    expected_answer=record["ground_truth"],
                    generated_answer=record["llm_answer"],
                    retrieved_contexts=record["retrieved_contexts"],
                    generation=RAGASEvalGeneration(
                        answer_relevancy=record["context_relevancy"],
                        faithfulness=record["faithfulness"],
                    ),
                    retrieval=RAGASEvalRetrieval(
                        context_recall=record["context_recall"],
                        context_precision=record["context_precision"],
                        answer_correctness=record["answer_precision"],
                        semantic_similarity=record["semantic_similarity"],
                    ),
                )
            )
        return results
