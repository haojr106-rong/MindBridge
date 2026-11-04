from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from app.agents.autonomous import (
    CoordinatorAgent,
    FINAL_FALLBACK_PROMPT,
    STATIC_SAFETY_RESPONSE,
    high_risk_prompt_check,
)
from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentTask,
    CollaborationBlackboard,
    PRIORITY_ORDER,
    TaskPriority,
    TaskStatus,
)
from app.agents.registry import AgentCapability, AgentRegistry
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage


class EventDrivenCoordinator:
    """Dependency-aware coordinator with per-agent budgets and a HIGH-risk safety gate."""

    def __init__(self, registry: AgentRegistry, coordinator_agent: CoordinatorAgent, settings: Settings):
        self.registry = registry
        self.coordinator_agent = coordinator_agent
        self.settings = settings
        self.max_rounds = int(getattr(settings, "agent_max_rounds", 8))
        self.final_min_confidence = float(getattr(settings, "agent_final_acceptance_min_confidence", 0.6))
        self.agent_limits = {
            "UnderstandingAgent": int(getattr(settings, "agent_max_understanding_runs", 1)),
            "ContextAgent": int(getattr(settings, "agent_max_context_runs", 1)),
            "ResponseAgent": int(getattr(settings, "agent_max_response_runs", 3)),
            "SafetyAgent": int(getattr(settings, "agent_max_safety_runs", 5)),
        }

    def run(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        board = self._ensure_root_task(board)
        claim_counts: dict[str, int] = defaultdict(int)
        for round_number in range(1, self.max_rounds + 1):
            board = board.append_event(
                AgentEvent(
                    type=AgentEventType.ROUND_STARTED,
                    actor=self.coordinator_agent.name,
                    message=f"round={round_number}",
                    metadata={"round": round_number},
                )
            )
            board = self._derive_missing_work(board)
            board = self._try_accept_final(board)
            if board.final_artifact_id or board.direct_response_text:
                return board

            candidates = self._claim_candidates(board, claim_counts)
            if not candidates:
                continue

            executed_high_context = any(
                candidate.agent.profile.name == "ContextAgent" and _risk_value(board) == RiskLevel.HIGH
                for _, candidate in candidates
            )
            board = self._execute_candidates(board, candidates, claim_counts)

            # HIGH still fits in eight global rounds: after the complete Context
            # artifact is produced, Response may run as a second wave in the
            # same round. Each agent still executes at most once in that round.
            if executed_high_context:
                board = self._derive_missing_work(board)
                response_candidates = [
                    item for item in self._claim_candidates(board, claim_counts)
                    if item[1].agent.profile.name == "ResponseAgent"
                ]
                board = self._execute_candidates(board, response_candidates, claim_counts)

            board = self._derive_missing_work(board)
            board = self._try_accept_final(board)
            if board.final_artifact_id or board.direct_response_text:
                return board

        if board.final_artifact_id or board.direct_response_text:
            return board
        return board.append_event(
            AgentEvent(
                type=AgentEventType.BUDGET_EXHAUSTED,
                actor=self.coordinator_agent.name,
                message="event-driven agent budget exhausted before final acceptance",
                metadata={"claimCounts": dict(claim_counts)},
            )
        )

    def _ensure_root_task(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        if board.tasks:
            return board
        root = self.coordinator_agent.root_task(board)
        return board.add_task(root).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=root.id, message=root.title)
        )

    def _derive_missing_work(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        board = self._ensure_task_for_missing_artifact(
            board, "intent", "task:understand", "Understand user turn", AgentCapability.UNDERSTANDING,
            TaskPriority.HIGH, bool(board.user_input), (), {"kind": "intent"},
        )
        board = self._ensure_task_for_missing_artifact(
            board, "risk", "task:assess-safety", "Assess safety risk", AgentCapability.SAFETY,
            TaskPriority.CRITICAL if _hard_high_risk(board.user_input) else TaskPriority.HIGH,
            bool(board.user_input), (), {"kind": "risk"},
        )
        if not board.has_artifact("intent") or not board.has_artifact("risk"):
            return board

        intent = _intent_value(board)
        risk = _risk_value(board)
        needs_context = not (intent == IntentType.CHAT and risk == RiskLevel.LOW)
        if needs_context:
            board = self._ensure_task_for_missing_artifact(
                board, "context", "task:gather-context", "Gather contextual evidence", AgentCapability.CONTEXT,
                TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.NORMAL,
                True, ("task:understand", "task:assess-safety"),
                {"kind": "context", "ragTimeoutSeconds": float(getattr(self.settings, "agent_context_rag_timeout_seconds", 3.0))},
            )

        response = board.latest_artifact("response_proposal")
        if response is None:
            response_dependencies = ["task:understand", "task:assess-safety"]
            if needs_context:
                response_dependencies.append("task:gather-context")
            board = self._ensure_task(
                board,
                AgentTask(
                    id="task:propose-response:1",
                    title="Propose candidate prompt 1",
                    description=board.user_input,
                    priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.HIGH,
                    required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
                    created_by=self.coordinator_agent.name,
                    depends_on=tuple(response_dependencies),
                    metadata={"kind": "response", "candidateAttempt": 1},
                ),
            )
            return board

        if risk != RiskLevel.HIGH:
            return board

        review = board.latest_artifact("safety_review")
        if review is None or review.metadata.get("responseArtifactId") != response.id:
            return self._ensure_task(
                board,
                AgentTask(
                    id=f"task:review-response:{response.id}",
                    title="Review candidate prompt safety",
                    description="Review the latest HIGH-risk prompt before final model delivery.",
                    priority=TaskPriority.CRITICAL,
                    required_capabilities=frozenset({AgentCapability.SAFETY.value}),
                    created_by=self.coordinator_agent.name,
                    depends_on=(response.task_id,),
                    metadata={"kind": "safety_review", "responseArtifactId": response.id},
                ),
            )

        if review.payload.get("approved"):
            return board

        attempt = int(response.metadata.get("candidateAttempt", 1))
        reason = str(review.payload.get("reason", "Safety review requested a targeted revision."))
        board = replace(board, latest_failure_reason=reason, revision_count=max(board.revision_count, attempt))
        if attempt < 3:
            next_attempt = attempt + 1
            return self._ensure_task(
                board,
                AgentTask(
                    id=f"task:propose-response:{next_attempt}",
                    title=f"Revise candidate prompt {next_attempt}",
                    description=reason,
                    priority=TaskPriority.CRITICAL,
                    required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
                    created_by=self.coordinator_agent.name,
                    depends_on=(review.task_id,),
                    metadata={
                        "kind": "response", "candidateAttempt": next_attempt,
                        "responseArtifactId": response.id, "lastFailureReason": reason,
                    },
                ),
            )
        if not response.metadata.get("fallback"):
            fallback = self._fallback_artifact(board, response.task_id)
            passed, missing = high_risk_prompt_check(fallback.payload["messages"])
            if not passed:
                return replace(
                    board,
                    workflow_status="FALLBACK_REJECTED",
                    direct_response_text=STATIC_SAFETY_RESPONSE,
                ).append_event(
                    AgentEvent(
                        type=AgentEventType.PROGRAM_CHECKED,
                        actor=self.coordinator_agent.name,
                        task_id=response.task_id,
                        message="fixed fallback prompt failed program check",
                        metadata={"passed": False, "missing": missing, "fallback": True},
                    )
                )
            board = board.add_artifact(fallback).append_event(
                AgentEvent(
                    type=AgentEventType.PROGRAM_CHECKED,
                    actor=self.coordinator_agent.name,
                    task_id=response.task_id,
                    artifact_id=fallback.id,
                    message="fixed fallback prompt passed program check",
                    metadata={"passed": True, "missing": [], "fallback": True},
                )
            )
            return board.append_event(
                AgentEvent(
                    type=AgentEventType.WORKFLOW_STATUS_CHANGED,
                    actor=self.coordinator_agent.name,
                    message="FALLBACK_PENDING_REVIEW",
                    metadata={"status": "FALLBACK_PENDING_REVIEW", "lastFailureReason": reason},
                )
            )

        rejected = replace(
            board, workflow_status="FALLBACK_REJECTED", direct_response_text=STATIC_SAFETY_RESPONSE,
            latest_failure_reason=reason,
        )
        return rejected.append_event(
            AgentEvent(
                type=AgentEventType.WORKFLOW_STATUS_CHANGED,
                actor=self.coordinator_agent.name,
                message="FALLBACK_REJECTED",
                metadata={"status": "FALLBACK_REJECTED", "lastFailureReason": reason},
            )
        )

    def _fallback_artifact(self, board: CollaborationBlackboard, task_id: str) -> AgentArtifact:
        return AgentArtifact(
            id=f"CoordinatorAgent:response_proposal:fallback:{board.turn_id[:10]}",
            owner=self.coordinator_agent.name,
            kind="response_proposal",
            payload={
                "messages": [
                    AiMessage(role="system", content=FINAL_FALLBACK_PROMPT),
                    AiMessage(role="user", content=board.model_input or board.user_input),
                ],
                "mode": "high_risk_fallback",
                "intent": _intent_value(board).value,
                "risk": RiskLevel.HIGH.value,
                "candidateAttempt": 4,
                "lastFailureReason": board.latest_failure_reason,
                "usedProgramFallback": True,
            },
            confidence=1.0,
            task_id=task_id,
            metadata={"candidateAttempt": 4, "fallback": True},
        )

    def _ensure_task_for_missing_artifact(
        self, board: CollaborationBlackboard, artifact_kind: str, task_id: str, title: str,
        capability: AgentCapability, priority: TaskPriority, condition: bool,
        depends_on: tuple[str, ...], metadata: dict,
    ) -> CollaborationBlackboard:
        if not condition or board.latest_artifact(artifact_kind) is not None:
            return board
        return self._ensure_task(
            board,
            AgentTask(
                id=task_id, title=title, description=board.user_input, priority=priority,
                required_capabilities=frozenset({capability.value}), created_by=self.coordinator_agent.name,
                depends_on=depends_on, metadata=metadata,
            ),
        )

    def _ensure_task(self, board: CollaborationBlackboard, task: AgentTask) -> CollaborationBlackboard:
        if task.id in board.tasks:
            return board
        return board.add_task(task).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=task.id, message=task.title)
        )

    def _dependencies_satisfied(self, task: AgentTask, board: CollaborationBlackboard) -> bool:
        for dependency in task.depends_on:
            dependency_task = board.tasks.get(dependency)
            if dependency_task is None or dependency_task.status != TaskStatus.CLOSED:
                return False
            expected_kind = dependency_task.metadata.get("kind")
            if expected_kind in {"intent", "risk", "context", "response", "safety_review"}:
                artifact_kind = "response_proposal" if expected_kind == "response" else expected_kind
                if not any(
                    artifact.kind == artifact_kind and artifact.task_id == dependency
                    for artifact in board.artifacts
                ):
                    return False
        return True

    def _claim_candidates(self, board: CollaborationBlackboard, claim_counts: dict[str, int]):
        task_candidates = []
        for task in board.open_tasks():
            if task.metadata.get("claimable") is False or not self._dependencies_satisfied(task, board):
                continue
            for candidate in self.registry.candidate_decisions_for(task, board):
                name = candidate.agent.profile.name
                if claim_counts[name] >= self.agent_limits.get(name, 1):
                    continue
                task_candidates.append((task, candidate))
        task_candidates.sort(
            key=lambda item: (
                PRIORITY_ORDER[item[0].priority], item[1].decision.confidence, item[1].agent.profile.name,
            ),
            reverse=True,
        )
        selected = []
        selected_agents = set()
        selected_tasks = set()
        for task, candidate in task_candidates:
            name = candidate.agent.profile.name
            if name in selected_agents or task.id in selected_tasks:
                continue
            selected.append((task, candidate))
            selected_agents.add(name)
            selected_tasks.add(task.id)
        return selected

    def _execute_candidates(self, board, candidates, claim_counts):
        if not candidates:
            return board
        claimed = []
        for task, candidate in candidates:
            current = board.tasks.get(task.id, task)
            board = board.update_task(current.claim(candidate.agent.profile.name)).append_event(
                AgentEvent(
                    type=AgentEventType.TASK_CLAIMED, actor=candidate.agent.profile.name, task_id=current.id,
                    message=candidate.decision.reason, metadata={"confidence": candidate.decision.confidence},
                )
            )
            claimed.append((current, candidate))
        with ThreadPoolExecutor(max_workers=len(claimed)) as executor:
            futures = [(task, candidate, executor.submit(candidate.agent.act, task, board)) for task, candidate in claimed]
            for task, candidate, future in futures:
                board = board.apply_turn_result(task, candidate.agent.profile.name, future.result())
                claim_counts[candidate.agent.profile.name] += 1
        return board

    def _try_accept_final(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        if board.final_artifact_id or board.direct_response_text:
            return board
        response = board.latest_artifact("response_proposal")
        if response is None or response.confidence < self.final_min_confidence:
            return board
        if _risk_value(board) == RiskLevel.HIGH:
            review = board.latest_artifact("safety_review")
            if review is None or review.metadata.get("responseArtifactId") != response.id or not review.payload.get("approved"):
                return board
            reason = "accepted after HIGH-risk prompt review"
        else:
            reason = "accepted without candidate review because risk is not HIGH"
        self.coordinator_agent.remember_acceptance(response.id, reason)
        return board.accept_final(response.id, self.coordinator_agent.name, reason)


def _intent_value(board: CollaborationBlackboard) -> IntentType:
    artifact = board.latest_artifact("intent")
    if artifact:
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT
    if _hard_high_risk(board.user_input):
        return IntentType.RISK
    return IntentType.CHAT


def _risk_value(board: CollaborationBlackboard) -> RiskLevel:
    order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
    highest = RiskLevel.LOW
    for artifact in board.artifacts_by_kind("risk"):
        try:
            risk = RiskLevel(str(artifact.payload.get("risk", RiskLevel.LOW.value)).upper())
        except ValueError:
            risk = RiskLevel.LOW
        if order[risk] > order[highest]:
            highest = risk
    if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
        return RiskLevel.HIGH
    return highest


def _hard_high_risk(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        word in lowered
        for word in ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"]
    )
