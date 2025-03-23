from collections import defaultdict
from typing import Dict, List, Any
from rag_meter.adapters.arize.arize_adapter import ArizeEvalMetric
from rag_meter.constants import ARIZE


class RAGEvaluationAggregator:
    """Aggregates evaluation results from multiple RAG frameworks."""

    def __init__(self, data: Dict[str, List[Any]]):
        """
        Initializes the aggregator.

        Args:
            data (Dict[str, List[Any]]): Raw evaluation results from different frameworks.
        """
        self.data = data
        self.aggregated = {"questions": [], "frameworks": []}
        self.framework_metrics = defaultdict(
            lambda: {"retrieval": defaultdict(list), "generation": defaultdict(list)}
        )
        self.seen_questions = set()

    def _compute_averages(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Computes average scores for a given framework."""
        return {
            metric: round(sum(values) / len(values), 3) if values else 0.0
            for metric, values in metrics.items()
        }

    def process_results(self) -> Dict[str, Any]:
        """Processes and aggregates the evaluation results."""
        for framework, results in self.data.items():
            for entry in results:
                question_id = entry.question

                # Store unique question details
                if question_id not in self.seen_questions:
                    self.seen_questions.add(question_id)
                    self.aggregated["questions"].append(
                        {
                            "question": entry.question,
                            "expected_answer": entry.expected_answer,
                            "generated_answer": entry.generated_answer,
                            "retrieved_contexts": getattr(
                                entry, "retrieved_contexts", []
                            ),
                        }
                    )

                # Handle Arize separately
                if framework == ARIZE:
                    for metric_name, metric_obj in vars(entry.generation).items():
                        if isinstance(metric_obj, ArizeEvalMetric):
                            self.framework_metrics[framework]["generation"][
                                metric_name
                            ].append(metric_obj.score)
                else:
                    # Collect retrieval and generation metrics
                    for metric, value in vars(entry.retrieval).items():
                        self.framework_metrics[framework]["retrieval"][metric].append(
                            value
                        )
                    for metric, value in vars(entry.generation).items():
                        self.framework_metrics[framework]["generation"][metric].append(
                            value
                        )

        # Compute and store averaged metrics
        for framework, metrics in self.framework_metrics.items():
            averaged_metrics = {
                "name": framework,
                "retrieval": self._compute_averages(metrics["retrieval"]),
                "generation": self._compute_averages(metrics["generation"]),
            }

            if framework == ARIZE:
                averaged_metrics["retrieval"] = {}  # Arize has no retrieval metrics

            self.aggregated["frameworks"].append(averaged_metrics)

        return self.aggregated
