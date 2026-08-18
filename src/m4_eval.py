from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])

        # Get aggregate scores
        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric_name in result:
                val = result[metric_name]
                scores[metric_name] = float(val) if hasattr(val, '__float__') else float(val)
            else:
                scores[metric_name] = 0.0

        # Get per-question results
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])) if row.get("contexts") else [],
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0)
            ))

        return {
            **scores,
            "per_question": per_question
        }
    except Exception as e:
        print(f"  [WARN] RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree mapping
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    }

    failures = []
    for result in eval_results:
        # Compute average of 4 metrics
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)

        # Find worst metric
        worst_metric = min(metrics.items(), key=lambda x: x[1])

        diagnosis, suggested_fix = diagnostic_tree.get(worst_metric[0], ("Unknown issue", "Review pipeline"))

        failures.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric[0],
            "score": worst_metric[1],
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    # Sort by avg_score ascending and take bottom_n
    failures.sort(key=lambda x: x["avg_score"])

    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
