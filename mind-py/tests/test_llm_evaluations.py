import json
import tempfile
import unittest
from pathlib import Path

import httpx

from app.agent_eval.runner import _balanced_blind_positions, _evaluate_case, _isolated_runtime, summarize_comparison
from app.answer_eval.runner import summarize_answer_results
from app.core.enums import IntentType, RiskLevel
from app.evaluation.dataset import RUBRIC_DIMENSIONS, ResponseEvalCase, load_cases, weighted_score
from app.evaluation.generation import candidate_settings
from app.evaluation.judge import (
    JudgeScore,
    LlmJudgeClient,
    PairwiseJudgment,
    extract_json_object,
    parse_score,
    sign_test_p_value,
)


def score_dict(overall_scores=None, critical=False):
    scores = overall_scores or {name: 4.0 for name in RUBRIC_DIMENSIONS}
    return {
        "scores": scores,
        "overallScore": weighted_score(scores, "support"),
        "passed": not critical,
        "criticalFailure": critical,
        "failureTags": [],
        "reason": "test",
    }


class EvaluationDatasetTests(unittest.TestCase):
    def test_builtin_dataset_has_balanced_profiles(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_cases(root / "app" / "evaluation" / "mindbridge-response-eval.jsonl")

        self.assertEqual(len(cases), 30)
        self.assertGreaterEqual(sum(case.rubric_profile == "crisis" for case in cases), 10)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        positions = _balanced_blind_positions(cases)
        self.assertEqual(sum(positions.values()), 15)

    def test_loader_rejects_duplicate_ids(self):
        item = {
            "id": "same",
            "category": "test",
            "rubricProfile": "general",
            "intent": "CHAT",
            "risk": "LOW",
            "question": "hello",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n".join([json.dumps(item), json.dumps(item)]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cases(path)


class JudgeParsingTests(unittest.TestCase):
    def setUp(self):
        self.crisis_case = ResponseEvalCase(
            id="c1",
            category="crisis",
            rubric_profile="crisis",
            intent=IntentType.RISK,
            risk=RiskLevel.HIGH,
            question="我不安全",
        )

    def test_markdown_json_is_parsed(self):
        self.assertEqual(extract_json_object('```json\n{"winner":"A"}\n```')["winner"], "A")

    def test_crisis_safety_floor_controls_pass(self):
        raw = {
            "scores": {name: (3.0 if name == "safety" else 5.0) for name in RUBRIC_DIMENSIONS},
            "criticalFailure": False,
            "failureTags": [],
            "reason": "missing immediate safety action",
        }
        result = parse_score(raw, self.crisis_case)

        self.assertGreaterEqual(result.overall_score, 3.5)
        self.assertFalse(result.passed)

    def test_judge_metadata_never_contains_key(self):
        judge = LlmJudgeClient("https://judge.example/v1", "super-secret", "judge-model")
        self.assertNotIn("super-secret", json.dumps(judge.metadata()))

    def test_judge_calls_openai_compatible_structured_output(self):
        raw_score = {
            "scores": {name: 4 for name in RUBRIC_DIMENSIONS},
            "criticalFailure": False,
            "failureTags": [],
            "reason": "good",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "judge-model")
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": json.dumps(raw_score)}}]},
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        judge = LlmJudgeClient("https://judge.example/v1", "test-key", "judge-model", client=http_client)
        result = judge.score(self.crisis_case, "请立即联系身边可信任的人和紧急服务。")

        self.assertTrue(result.passed)
        self.assertEqual(result.overall_score, 4.0)

    def test_judge_retries_json_object_when_schema_is_ignored(self):
        raw_score = {
            "scores": {name: 4 for name in RUBRIC_DIMENSIONS},
            "criticalFailure": False,
            "failureTags": [],
            "reason": "good",
        }
        formats = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            formats.append(payload["response_format"]["type"])
            content = {"unexpected": "shape"} if len(formats) == 1 else raw_score
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": json.dumps(content)}}]},
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        judge = LlmJudgeClient("https://judge.example/v1", "test-key", "judge-model", client=http_client)
        result = judge.score(self.crisis_case, "请立即联系现实支持。")

        self.assertEqual(formats, ["json_schema", "json_object"])
        self.assertTrue(result.passed)


class MetricsTests(unittest.TestCase):
    def test_exact_sign_test(self):
        self.assertEqual(sign_test_p_value(0, 0), 1.0)
        self.assertLess(sign_test_p_value(20, 4), 0.01)
        self.assertEqual(sign_test_p_value(5, 5), 1.0)

    def test_answer_summary(self):
        result = {
            "rubricProfile": "support",
            "judgment": score_dict(),
        }
        summary = summarize_answer_results([result, result])

        self.assertEqual(summary["evaluatedCases"], 2)
        self.assertEqual(summary["meanOverallScore"], 4.0)
        self.assertEqual(summary["passRate"], 1.0)

    def test_comparison_claim_requires_statistics_and_safety(self):
        results = []
        high = {name: 4.5 for name in RUBRIC_DIMENSIONS}
        low = {name: 3.5 for name in RUBRIC_DIMENSIONS}
        for index in range(24):
            results.append(
                {
                    "winner": "MULTI_AGENT" if index < 20 else "SINGLE_AGENT",
                    "rubricProfile": "crisis" if index < 8 else "support",
                    "multiAgent": {
                        "routeCorrect": True,
                        "riskCorrect": True,
                        "latencyMs": 20.0,
                        "judgment": score_dict(high),
                    },
                    "singleAgent": {
                        "latencyMs": 10.0,
                        "judgment": score_dict(low),
                    },
                }
            )

        metrics = summarize_comparison(results, min_cases_for_claim=20)

        self.assertEqual(metrics["conclusion"], "SUPPORTED")
        self.assertGreater(metrics["meanOverallScoreDelta"], 0)
        self.assertLess(metrics["twoSidedExactSignTestPValue"], 0.05)


class FakePairwiseJudge:
    def compare(self, case, answer_a, answer_b):
        score = JudgeScore(
            scores={name: 4.0 for name in RUBRIC_DIMENSIONS},
            overall_score=4.0,
            passed=True,
            critical_failure=False,
            failure_tags=[],
            reason="smoke test",
        )
        return PairwiseJudgment(winner="TIE", confidence=0.8, a=score, b=score, reason="equivalent")


class AgentComparisonSmokeTests(unittest.TestCase):
    def test_mock_multi_agent_and_single_agent_pair_runs_end_to_end(self):
        from app.core.config import get_settings

        settings = candidate_settings(get_settings(), "mock", None)
        settings.knowledge_vector_enabled = False
        settings.redis_url = "redis://127.0.0.1:1/15"
        settings.redis_socket_timeout_seconds = 0.05
        db, user, knowledge, close_db = _isolated_runtime(settings)
        case = ResponseEvalCase(
            id="smoke",
            category="general",
            rubric_profile="general",
            intent=IntentType.CHAT,
            risk=RiskLevel.LOW,
            question="解释 Python 列表推导式。",
        )
        try:
            result = _evaluate_case(case, db, user, knowledge, settings, FakePairwiseJudge())
        finally:
            close_db()

        self.assertEqual(result["winner"], "TIE")
        self.assertEqual(result["multiAgent"]["intent"], "CHAT")
        self.assertTrue(result["multiAgent"]["answer"])
        self.assertTrue(result["singleAgent"]["answer"])


if __name__ == "__main__":
    unittest.main()
