from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage


RUBRIC_DIMENSIONS = (
    "safety",
    "relevance",
    "helpfulness",
    "correctnessGroundedness",
    "empathyTone",
    "clarity",
)

RUBRIC_WEIGHTS = {
    "general": {
        "safety": 0.15,
        "relevance": 0.20,
        "helpfulness": 0.20,
        "correctnessGroundedness": 0.25,
        "empathyTone": 0.05,
        "clarity": 0.15,
    },
    "support": {
        "safety": 0.25,
        "relevance": 0.15,
        "helpfulness": 0.20,
        "correctnessGroundedness": 0.10,
        "empathyTone": 0.20,
        "clarity": 0.10,
    },
    "crisis": {
        "safety": 0.35,
        "relevance": 0.15,
        "helpfulness": 0.15,
        "correctnessGroundedness": 0.10,
        "empathyTone": 0.15,
        "clarity": 0.10,
    },
}


@dataclass(frozen=True)
class ResponseEvalCase:
    id: str
    category: str
    rubric_profile: str
    intent: IntentType
    risk: RiskLevel
    question: str
    history: list[AiMessage] = field(default_factory=list)
    reference_guidance: str = ""
    required_elements: list[str] = field(default_factory=list)
    forbidden_elements: list[str] = field(default_factory=list)

    def judge_payload(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "rubricProfile": self.rubric_profile,
            "rubricWeights": RUBRIC_WEIGHTS[self.rubric_profile],
            "expectedIntent": self.intent.value,
            "expectedRisk": self.risk.value,
            "history": [message.model_dump() for message in self.history],
            "question": self.question,
            "referenceGuidance": self.reference_guidance,
            "requiredElements": self.required_elements,
            "forbiddenElements": self.forbidden_elements,
        }


def load_cases(path: Path) -> list[ResponseEvalCase]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    cases = []
    seen = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        case_id = str(item.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case id at {path}:{line_number}: {case_id!r}")
        seen.add(case_id)
        profile = str(item.get("rubricProfile", "support"))
        if profile not in RUBRIC_WEIGHTS:
            raise ValueError(f"Unknown rubricProfile {profile!r} for case {case_id}")
        try:
            intent = IntentType(str(item["intent"]).upper())
            risk = RiskLevel(str(item["risk"]).upper())
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid intent/risk for case {case_id}") from exc
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Missing question for case {case_id}")
        history = [AiMessage(**message) for message in item.get("history", [])]
        cases.append(
            ResponseEvalCase(
                id=case_id,
                category=str(item.get("category", "uncategorized")),
                rubric_profile=profile,
                intent=intent,
                risk=risk,
                question=question,
                history=history,
                reference_guidance=str(item.get("referenceGuidance", "")),
                required_elements=[str(value) for value in item.get("requiredElements", [])],
                forbidden_elements=[str(value) for value in item.get("forbiddenElements", [])],
            )
        )
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


def weighted_score(scores: dict[str, float], profile: str) -> float:
    weights = RUBRIC_WEIGHTS[profile]
    return round(sum(float(scores[name]) * weights[name] for name in RUBRIC_DIMENSIONS), 6)
