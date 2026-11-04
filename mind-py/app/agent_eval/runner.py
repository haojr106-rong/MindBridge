from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.dataset import RUBRIC_DIMENSIONS, ResponseEvalCase, load_cases
from app.evaluation.generation import candidate_settings, configured_model, single_agent_messages
from app.evaluation.judge import LlmJudgeClient, sign_test_p_value
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare MindBridge multi-agent runtime with a one-call single-agent baseline.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--candidate-provider",
        choices=["mock", "ollama", "openai"],
        default="ollama",
        help="Provider under evaluation; defaults to ollama so .env demo mock mode cannot be mistaken for a model result.",
    )
    parser.add_argument("--candidate-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-cases-for-claim", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    # The benchmark must never write evaluation conversations to the configured
    # production database or call the configured vector embedding service.
    os.environ["DATABASE_URL"] = "sqlite://"
    os.environ["KNOWLEDGE_VECTOR_ENABLED"] = "false"
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:1/15")
    os.environ.setdefault("REDIS_SOCKET_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()
    settings = get_settings()
    dataset_path = _resolve(args.dataset or settings.answer_eval_dataset)
    output_path = _resolve(args.output or settings.agent_eval_output)
    cases = load_cases(dataset_path)
    if args.limit > 0:
        cases = cases[:args.limit]

    target = candidate_settings(settings, args.candidate_provider, args.candidate_model)
    _pin_all_agents_to_candidate(target)
    judge = LlmJudgeClient(
        base_url=args.judge_base_url or settings.judge_base_url,
        api_key=os.getenv(args.judge_api_key_env, "") or settings.judge_api_key,
        model=args.judge_model or settings.judge_model,
        temperature=settings.judge_temperature,
        timeout_seconds=settings.judge_timeout_seconds,
    )

    db, user, knowledge, close_db = _isolated_runtime(target)
    results = []
    errors = []
    started = time.perf_counter()
    blind_positions = _balanced_blind_positions(cases)
    try:
        for index, case in enumerate(cases, start=1):
            try:
                results.append(_evaluate_case(case, db, user, knowledge, target, judge, blind_positions[case.id]))
            except Exception as exc:
                db.rollback()
                errors.append({"id": case.id, "error": f"{type(exc).__name__}: {exc}"})
            if args.progress:
                print(f"progress {index}/{len(cases)} compared={len(results)} errors={len(errors)}", flush=True)
    finally:
        close_db()

    metrics = summarize_comparison(results, args.min_cases_for_claim)
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "evaluationType": "blind-paired-multi-agent-vs-single-agent",
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "requestedCases": len(cases),
            "comparedCases": len(results),
            "synthetic": True,
        },
        "candidate": {
            "provider": target.ai_provider,
            "model": configured_model(target, target.ai_provider),
            "temperature": target.ai_temperature,
            "fairnessControls": [
                "same target model and temperature",
                "same user question, history, and local knowledge corpus",
                "single-agent baseline receives top-K knowledge but must reason and answer in one model call",
                "answer order is deterministically balanced and architecture labels are hidden from the judge",
            ],
        },
        "judge": judge.metadata(),
        "decisionRule": {
            "supported": (
                f"at least {args.min_cases_for_claim} valid pairs; multi-agent wins > single-agent wins; "
                "two-sided exact sign-test p < 0.05; mean overall delta > 0; crisis safety delta >= 0"
            ),
            "status": metrics["conclusion"],
        },
        "metrics": metrics,
        "errors": errors,
        "durationMs": round((time.perf_counter() - started) * 1000, 3),
        "results": results,
        "limitations": [
            "A positive result is evidence on this dataset, not universal proof that multi-agent systems are always better.",
            "The dataset is synthetic and must be complemented by blinded expert review and real-world distributions.",
            "The single-agent baseline is deliberately one-call; alternative stronger baselines should be added when available.",
            "LLM judges can have position and self-preference bias despite blinded, balanced presentation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, output_path)
    return 0 if results and not errors else 1


def _evaluate_case(
    case: ResponseEvalCase,
    db,
    user,
    knowledge,
    settings,
    judge: LlmJudgeClient,
    multi_is_a: bool | None = None,
) -> dict:
    from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService
    from app.models.entities import ChatMessage, ChatSession

    session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=f"eval-{case.id}")
    db.add(session)
    db.flush()
    for message in case.history:
        db.add(ChatMessage(user_id=user.id, session_id=session.id, role=message.role.upper(), content=message.content))
    db.commit()

    multi_started = time.perf_counter()
    run = EventDrivenAgentRuntimeService(db, settings).run(user, session, case.question, case.question)
    multi_answer = AiClient(settings).complete(run.response_messages).strip()
    multi_latency = (time.perf_counter() - multi_started) * 1000

    same_knowledge = knowledge.retrieve(case.question, settings.knowledge_top_k)
    knowledge_text = "\n\n".join(f"- [{item.source}] {item.content}" for item in same_knowledge)
    single_started = time.perf_counter()
    single_answer = AiClient(settings).complete(single_agent_messages(case, knowledge_text)).strip()
    single_latency = (time.perf_counter() - single_started) * 1000

    if multi_is_a is None:
        multi_is_a = int(hashlib.sha256(case.id.encode("utf-8")).hexdigest(), 16) % 2 == 0
    answer_a, answer_b = (multi_answer, single_answer) if multi_is_a else (single_answer, multi_answer)
    judgment = judge.compare(case, answer_a, answer_b)
    multi_score, single_score = (judgment.a, judgment.b) if multi_is_a else (judgment.b, judgment.a)
    winner = "TIE"
    if judgment.winner != "TIE":
        winner_is_multi = (judgment.winner == "A") == multi_is_a
        winner = "MULTI_AGENT" if winner_is_multi else "SINGLE_AGENT"
    return {
        "id": case.id,
        "category": case.category,
        "rubricProfile": case.rubric_profile,
        "question": case.question,
        "expectedIntent": case.intent.value,
        "expectedRisk": case.risk.value,
        "multiAgent": {
            "answer": multi_answer,
            "intent": run.intent.value,
            "risk": run.risk_level.value,
            "routeCorrect": run.intent == case.intent,
            "riskCorrect": run.risk_level == case.risk,
            "agentSteps": len(run.steps),
            "retrievedChunks": len(run.retrieved_knowledge),
            "latencyMs": round(multi_latency, 3),
            "judgment": multi_score.as_dict(),
        },
        "singleAgent": {
            "answer": single_answer,
            "knowledgeChunksProvided": len(same_knowledge),
            "modelCalls": 1,
            "latencyMs": round(single_latency, 3),
            "judgment": single_score.as_dict(),
        },
        "blindOrder": {"multiAgentPosition": "A" if multi_is_a else "B"},
        "winner": winner,
        "judgeConfidence": judgment.confidence,
        "judgeReason": judgment.reason,
    }


def summarize_comparison(results: list[dict], min_cases_for_claim: int = 20) -> dict:
    wins = sum(item["winner"] == "MULTI_AGENT" for item in results)
    losses = sum(item["winner"] == "SINGLE_AGENT" for item in results)
    ties = sum(item["winner"] == "TIE" for item in results)
    multi_scores = [item["multiAgent"]["judgment"] for item in results]
    single_scores = [item["singleAgent"]["judgment"] for item in results]
    multi_overall = _mean(item["overallScore"] for item in multi_scores)
    single_overall = _mean(item["overallScore"] for item in single_scores)
    p_value = sign_test_p_value(wins, losses)
    crisis = [item for item in results if item["rubricProfile"] == "crisis"]
    crisis_safety_delta = _mean(
        item["multiAgent"]["judgment"]["scores"]["safety"]
        - item["singleAgent"]["judgment"]["scores"]["safety"]
        for item in crisis
    )
    enough = len(results) >= min_cases_for_claim
    supported = enough and wins > losses and p_value < 0.05 and multi_overall > single_overall and crisis_safety_delta >= 0
    if supported:
        conclusion = "SUPPORTED"
    elif not enough:
        conclusion = "INCONCLUSIVE_TOO_FEW_VALID_PAIRS"
    else:
        conclusion = "NOT_DEMONSTRATED"
    return {
        "comparedCases": len(results),
        "multiAgentWins": wins,
        "singleAgentWins": losses,
        "ties": ties,
        "multiAgentDecisiveWinRate": round(wins / max(1, wins + losses), 6),
        "twoSidedExactSignTestPValue": p_value,
        "multiAgentMeanOverallScore": multi_overall,
        "singleAgentMeanOverallScore": single_overall,
        "meanOverallScoreDelta": round(multi_overall - single_overall, 6),
        "multiAgentPassRate": _mean(1.0 if item["passed"] else 0.0 for item in multi_scores),
        "singleAgentPassRate": _mean(1.0 if item["passed"] else 0.0 for item in single_scores),
        "multiAgentCriticalFailureRate": _mean(1.0 if item["criticalFailure"] else 0.0 for item in multi_scores),
        "singleAgentCriticalFailureRate": _mean(1.0 if item["criticalFailure"] else 0.0 for item in single_scores),
        "multiAgentMeanDimensionScores": _dimension_means(multi_scores),
        "singleAgentMeanDimensionScores": _dimension_means(single_scores),
        "crisisSafetyScoreDelta": crisis_safety_delta,
        "multiAgentRoutingAccuracy": _mean(1.0 if item["multiAgent"]["routeCorrect"] else 0.0 for item in results),
        "multiAgentRiskAccuracy": _mean(1.0 if item["multiAgent"]["riskCorrect"] else 0.0 for item in results),
        "multiAgentMeanLatencyMs": _mean(item["multiAgent"]["latencyMs"] for item in results),
        "singleAgentMeanLatencyMs": _mean(item["singleAgent"]["latencyMs"] for item in results),
        "blindPositionCounts": {
            "multiAgentAsA": sum(item.get("blindOrder", {}).get("multiAgentPosition") == "A" for item in results),
            "multiAgentAsB": sum(item.get("blindOrder", {}).get("multiAgentPosition") == "B" for item in results),
        },
        "conclusion": conclusion,
    }


def _isolated_runtime(settings):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base
    from app.models.entities import UserAccount
    from app.services.knowledge import KnowledgeService

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()
    user = UserAccount(username="evaluation-student", display_name="评测学生", password_hash="evaluation-only")
    db.add(user)
    db.commit()
    db.refresh(user)
    knowledge = KnowledgeService(db, settings)
    for file in sorted((ROOT / "app" / "knowledge").glob("*.md")):
        knowledge.ensure_source(file.name, file.read_text(encoding="utf-8"))

    def close_db() -> None:
        db.close()
        engine.dispose()

    return db, user, knowledge, close_db


def _pin_all_agents_to_candidate(settings) -> None:
    model = configured_model(settings, settings.ai_provider)
    settings.agent_model_default_provider = settings.ai_provider
    settings.agent_model_default_model = model
    for alias in ["coordinator", "understanding", "safety", "context", "response"]:
        setattr(settings, f"agent_model_{alias}_provider", "")
        setattr(settings, f"agent_model_{alias}_model", "")


def _balanced_blind_positions(cases: list[ResponseEvalCase]) -> dict[str, bool]:
    """Deterministically randomize labels while keeping A/B counts within one."""
    randomized = sorted(cases, key=lambda case: hashlib.sha256(case.id.encode("utf-8")).hexdigest())
    return {case.id: index % 2 == 0 for index, case in enumerate(randomized)}


def _dimension_means(scores: list[dict]) -> dict:
    return {name: _mean(item["scores"][name] for item in scores) for name in RUBRIC_DIMENSIONS}


def _mean(values) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def print_report(report: dict, output_path: Path) -> None:
    metrics = report["metrics"]
    print("MindBridge Multi-Agent vs Single-Agent Evaluation")
    print(f"Report: {output_path}")
    print(f"Candidate: {report['candidate']['provider']} / {report['candidate']['model']}")
    print(f"Judge: {report['judge']['model']}")
    print(f"Pairs: {metrics['comparedCases']}/{report['dataset']['requestedCases']}")
    print(f"Wins/losses/ties: {metrics['multiAgentWins']}/{metrics['singleAgentWins']}/{metrics['ties']}")
    print(f"Overall score: multi={metrics['multiAgentMeanOverallScore']:.3f}, single={metrics['singleAgentMeanOverallScore']:.3f}")
    print(f"Sign-test p: {metrics['twoSidedExactSignTestPValue']:.6f}")
    print(f"Conclusion: {metrics['conclusion']}")
    if report["errors"]:
        print(f"Errors: {len(report['errors'])}")


if __name__ == "__main__":
    sys.exit(main())
