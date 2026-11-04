from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.enums import EmotionLabel, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates
from app.services.assessment import (
    PsychologicalAssessmentService,
    PsychologyAssessment,
    risk_from_score,
    risk_order,
    score_for_emotion,
)


LABELS = [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "app" / "risk_eval" / "mindbridge-risk-eval-100.jsonl"
DEFAULT_OUTPUT = ROOT / "target" / "risk-eval-report.json"


class TrackedAiClient:
    """Keep provider failures visible even though assessment has a safe fallback."""

    def __init__(self, client: AiClient):
        self.client = client
        self.calls = 0
        self.failures = 0

    def complete(self, messages: list[AiMessage]) -> str:
        self.calls += 1
        try:
            return self.client.complete(messages)
        except Exception:
            self.failures += 1
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MindBridge LOW/MEDIUM/HIGH risk classification.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="UTF-8 JSONL evaluation dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON report output path.")
    parser.add_argument(
        "--provider",
        choices=["mock", "ollama", "openai"],
        default=None,
        help="AI provider override. Omit to use the project's configured provider.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional model temperature override.")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "model-only"],
        default="pipeline",
        help="Evaluate the production assessment pipeline or send every case directly to the model.",
    )
    parser.add_argument("--progress", action="store_true", help="Print progress while evaluating cases.")
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.provider:
        settings.ai_provider = args.provider
    if args.temperature is not None:
        settings.ai_temperature = args.temperature

    dataset_path = Path(args.dataset).resolve()
    cases = load_cases(dataset_path)
    tracked_ai = TrackedAiClient(AiClient(settings))
    service = PsychologicalAssessmentService(tracked_ai)

    predictions = []
    model_parse_failures = 0
    started = time.perf_counter()
    heuristic_summaries = {
        "检测到低落或抑郁相关表达",
        "检测到焦虑或压力相关表达",
        "未检测到明显风险信号",
    }
    for index, case in enumerate(cases, start=1):
        history = [AiMessage(**message) for message in case.get("history", [])]
        case_started = time.perf_counter()
        try:
            result = (
                service.assess(case["text"], history)
                if args.mode == "pipeline"
                else assess_with_model_only(tracked_ai, case["text"], history)
            )
            predicted_risk = result.risk.value
            predicted_emotion = result.emotion.value
            confidence = result.confidence
            used_fallback = result.summary in heuristic_summaries
        except Exception:
            if args.mode != "model-only":
                raise
            model_parse_failures += 1
            predicted_risk = "INVALID"
            predicted_emotion = "INVALID"
            confidence = 0.0
            used_fallback = False
        predictions.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expectedRisk": case["expectedRisk"],
                "predictedRisk": predicted_risk,
                "predictedEmotion": predicted_emotion,
                "confidence": confidence,
                "usedAssessmentFallback": used_fallback,
                "correct": predicted_risk == case["expectedRisk"],
                "latencyMs": round((time.perf_counter() - case_started) * 1000, 3),
            }
        )
        if args.progress and (index == 1 or index % 10 == 0 or index == len(cases)):
            print(
                f"progress {index}/{len(cases)} provider_calls={tracked_ai.calls} "
                f"provider_failures={tracked_ai.failures}",
                flush=True,
            )

    metrics = classification_metrics(
        [item["expectedRisk"] for item in predictions],
        [item["predictedRisk"] for item in predictions],
    )
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "totalCases": len(cases),
            "synthetic": True,
            "containsRealPersonalData": False,
            "labelDistribution": dict(Counter(case["expectedRisk"] for case in cases)),
            "categoryDistribution": dict(Counter(case["category"] for case in cases)),
        },
        "evaluation": {
            "provider": settings.ai_provider,
            "model": provider_model(settings),
            "pipeline": "PsychologicalAssessmentService",
            "mode": args.mode,
            "providerCalls": tracked_ai.calls,
            "providerFailures": tracked_ai.failures,
            "modelParseFailures": model_parse_failures,
            "assessmentFallbacks": sum(item["usedAssessmentFallback"] for item in predictions),
            "temperature": settings.ai_temperature,
            "durationMs": round((time.perf_counter() - started) * 1000, 3),
            "metricDefinitions": {
                "highRiskRecall": "actual HIGH predicted HIGH / all actual HIGH",
                "highRiskMissRate": "actual HIGH predicted non-HIGH / all actual HIGH",
                "highRiskFalsePositiveRate": "actual non-HIGH predicted HIGH / all actual non-HIGH",
                "highRiskFalseDiscoveryRate": "actual non-HIGH predicted HIGH / all predicted HIGH",
                "confusionMatrix": "rows are actual labels; columns are predicted labels",
            },
        },
        "metrics": metrics,
        "categoryBreakdown": category_breakdown(predictions),
        "predictions": predictions,
        "limitations": [
            "All 100 cases are synthetic and were not clinically annotated.",
            "This benchmark measures single-run classification behavior, not clinical efficacy.",
            "Mock-provider results measure the current deterministic demo pipeline, not a real LLM.",
        ],
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, output_path)
    return 0


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        case = json.loads(raw)
        missing = {"id", "category", "text", "expectedRisk"} - set(case)
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        if case["expectedRisk"] not in LABELS:
            raise ValueError(f"line {line_number} has invalid expectedRisk: {case['expectedRisk']}")
        cases.append(case)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate ids")
    return cases


def classification_metrics(actual: list[str], predicted: list[str]) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must have equal, non-zero length")

    predicted_labels = LABELS + sorted(set(predicted) - set(LABELS))
    matrix = {
        actual_label: {
            predicted_label: sum(
                1
                for truth, guess in zip(actual, predicted)
                if truth == actual_label and guess == predicted_label
            )
            for predicted_label in predicted_labels
        }
        for actual_label in LABELS
    }
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in predicted_labels if other != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(matrix[label].values()),
            "truePositive": tp,
            "falsePositive": fp,
            "falseNegative": fn,
        }

    high_tp = matrix[RiskLevel.HIGH.value][RiskLevel.HIGH.value]
    high_fn = sum(
        matrix[RiskLevel.HIGH.value][label]
        for label in predicted_labels
        if label != RiskLevel.HIGH.value
    )
    high_fp = sum(matrix[label][RiskLevel.HIGH.value] for label in LABELS if label != RiskLevel.HIGH.value)
    actual_non_high = sum(1 for label in actual if label != RiskLevel.HIGH.value)
    accuracy = safe_divide(sum(matrix[label][label] for label in LABELS), len(actual))

    return {
        "accuracy": round(accuracy, 6),
        "macroF1": round(sum(per_class[label]["f1"] for label in LABELS) / len(LABELS), 6),
        "highRiskRecall": round(safe_divide(high_tp, high_tp + high_fn), 6),
        "highRiskMissRate": round(safe_divide(high_fn, high_tp + high_fn), 6),
        "highRiskFalsePositiveRate": round(safe_divide(high_fp, actual_non_high), 6),
        "highRiskFalseDiscoveryRate": round(safe_divide(high_fp, high_tp + high_fp), 6),
        "perClass": per_class,
        "confusionMatrix": {
            "labels": LABELS,
            "predictedLabels": predicted_labels,
            "rowsActualColumnsPredicted": matrix,
        },
    }


def category_breakdown(predictions: list[dict]) -> dict:
    breakdown = {}
    for category in sorted({item["category"] for item in predictions}):
        items = [item for item in predictions if item["category"] == category]
        breakdown[category] = {
            "total": len(items),
            "correct": sum(item["correct"] for item in items),
            "accuracy": round(safe_divide(sum(item["correct"] for item in items), len(items)), 6),
            "expected": dict(Counter(item["expectedRisk"] for item in items)),
            "predicted": dict(Counter(item["predictedRisk"] for item in items)),
        }
    return breakdown


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def assess_with_model_only(ai: TrackedAiClient, text: str, history: list[AiMessage]) -> PsychologyAssessment:
    raw = ai.complete(PromptTemplates.psychology_prompt(history, text))
    start = raw.find("{")
    end = raw.rfind("}")
    data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)
    emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
    score = float(data.get("emotionScore", score_for_emotion(emotion)))
    risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
    score_risk = risk_from_score(score)
    if risk_order(score_risk) > risk_order(risk):
        risk = score_risk
    if emotion == EmotionLabel.HIGH_RISK:
        risk = RiskLevel.HIGH
    return PsychologyAssessment(emotion, score, risk, confidence, data.get("summary", "模型评估结果"))


def provider_model(settings) -> str:
    if settings.ai_provider == "ollama":
        return settings.ollama_model
    if settings.ai_provider == "openai":
        return settings.openai_model
    return "mock"


def print_report(report: dict, output_path: Path) -> None:
    metrics = report["metrics"]
    print("MindBridge Risk Evaluation")
    print(f"Provider: {report['evaluation']['provider']} / {report['evaluation']['model']}")
    print(f"Cases: {report['dataset']['totalCases']}")
    print(f"HIGH_RISK recall: {metrics['highRiskRecall']:.2%}")
    print(f"HIGH_RISK miss rate: {metrics['highRiskMissRate']:.2%}")
    print(f"HIGH_RISK false-positive rate: {metrics['highRiskFalsePositiveRate']:.2%}")
    print(f"HIGH_RISK false-discovery rate: {metrics['highRiskFalseDiscoveryRate']:.2%}")
    print("")
    print("Per-class metrics")
    for label in LABELS:
        item = metrics["perClass"][label]
        print(
            f"{label:6} precision={item['precision']:.2%} "
            f"recall={item['recall']:.2%} f1={item['f1']:.2%} support={item['support']}"
        )
    print("")
    print("Confusion matrix (actual rows, predicted columns)")
    predicted_labels = metrics["confusionMatrix"].get("predictedLabels", LABELS)
    print("actual\\pred  " + "  ".join(f"{label:>7}" for label in predicted_labels))
    matrix = metrics["confusionMatrix"]["rowsActualColumnsPredicted"]
    for actual_label in LABELS:
        print(
            f"{actual_label:>11}  "
            + "  ".join(f"{matrix[actual_label][label]:>7}" for label in predicted_labels)
        )
    print("")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
