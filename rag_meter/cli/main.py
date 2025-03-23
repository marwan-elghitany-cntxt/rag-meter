import argparse
import json
import asyncio
from rag_meter.evaluation import RAGMeter


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Meter: A unified evaluation framework for RAG models."
    )

    # CLI Arguments
    parser.add_argument(
        "--contexts",
        nargs="+",
        required=True,
        help="List of retrieved contexts (separated by spaces).",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        required=True,
        help="List of corresponding questions for the contexts.",
    )
    parser.add_argument(
        "--answers",
        nargs="+",
        required=True,
        help="List of expected answers to evaluate.",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["ragas", "deepeval", "trulens", "arize"],
        default=["ragas", "deepeval"],
        help="Evaluation frameworks to use (default: ragas, deepeval).",
    )
    parser.add_argument(
        "--output", choices=["json", "text"], default="text", help="Output format."
    )

    args = parser.parse_args()

    # Create an instance of RAGMeter
    evaluator = RAGMeter(
        contexts=args.contexts,
        questions=args.questions,
        answers=args.answers,
        frameworks=args.frameworks,
    )

    # Run evaluations asynchronously
    results = asyncio.run(evaluator.multi_evaluation())

    # Output results
    if args.output == "json":
        print(json.dumps(results, indent=4))
    else:
        print("Evaluation Results:")
        for fw, res in results.items():
            print(f"\n[{fw.upper()}] Result: {res}")


if __name__ == "__main__":
    main()
