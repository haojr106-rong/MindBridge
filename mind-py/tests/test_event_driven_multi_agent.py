import unittest
import time
from types import SimpleNamespace

from app.agents.coordinator import EventDrivenCoordinator
from app.agents.autonomous import ContextAgent, HIGH_RISK_GUIDANCE_BLOCK, ResponseAgent
from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
    TaskPriority,
    TaskStatus,
)
from app.agents.registry import AgentCapability, AgentDecision, AgentProfile, AgentRegistry
from app.services.agent_models import AgentModelRegistry
from app.schemas.dtos import AiMessage


class EventDrivenProtocolTests(unittest.TestCase):
    def test_blackboard_is_append_only(self):
        board = CollaborationBlackboard(turn_id="t1")
        task = AgentTask(id="task-1", title="Resolve")
        event = AgentEvent(type=AgentEventType.TASK_CREATED, actor="CoordinatorAgent", task_id=task.id)

        updated = board.add_task(task).append_event(event)

        self.assertEqual(board.tasks, {})
        self.assertEqual(updated.tasks["task-1"].status, TaskStatus.OPEN)
        self.assertEqual(updated.events[0].actor, "CoordinatorAgent")

    def test_artifacts_do_not_overwrite_each_other(self):
        first = AgentArtifact(id="a1", owner="UnderstandingAgent", kind="intent", payload={"intent": "CHAT"})
        second = AgentArtifact(id="a2", owner="SafetyAgent", kind="risk", payload={"risk": "HIGH"})
        board = CollaborationBlackboard(turn_id="t1").add_artifact(first).add_artifact(second)

        self.assertEqual(len(board.artifacts), 2)
        self.assertEqual(board.artifacts_by_kind("risk")[0].payload["risk"], "HIGH")

    def test_latest_response_candidate_replaces_old_content(self):
        first = AgentArtifact(id="r1", owner="ResponseAgent", kind="response_proposal", payload={"value": 1})
        second = AgentArtifact(id="r2", owner="ResponseAgent", kind="response_proposal", payload={"value": 2})

        board = CollaborationBlackboard(turn_id="t1").add_artifact(first).add_artifact(second)

        self.assertEqual([artifact.id for artifact in board.artifacts_by_kind("response_proposal")], ["r2"])


class DemoAgent:
    def __init__(self, name, capability, confidence):
        self.profile = AgentProfile(
            name=name,
            capabilities=frozenset({capability}),
            system_prompt=f"{name} prompt",
            memory_policy="private",
            model_profile=name,
        )
        self.confidence = confidence

    def decide(self, task, board):
        return AgentDecision(True, self.confidence, f"{self.profile.name} claims")

    def act(self, task, board):
        return AgentTurnResult(
            artifacts=(
                AgentArtifact(
                    id=f"{self.profile.name}:artifact",
                    owner=self.profile.name,
                    kind="demo",
                    payload={"agent": self.profile.name},
                    task_id=task.id,
                ),
            )
        )


class RegistryAndCoordinatorTests(unittest.TestCase):
    def test_registry_sorts_by_claim_confidence_not_list_order(self):
        low = DemoAgent("LowConfidenceAgent", AgentCapability.UNDERSTANDING, 0.2)
        high = DemoAgent("HighConfidenceAgent", AgentCapability.UNDERSTANDING, 0.9)
        registry = AgentRegistry([low, high])
        task = AgentTask(id="task-1", title="Understand", required_capabilities=frozenset({AgentCapability.UNDERSTANDING.value}))

        candidates = registry.candidates_for(task, CollaborationBlackboard(turn_id="t1"))

        self.assertEqual([agent.profile.name for agent in candidates], ["HighConfidenceAgent", "LowConfidenceAgent"])

    def test_coordinator_uses_claims_for_open_tasks(self):
        settings = SimpleNamespace(
            agent_max_rounds=1,
            agent_max_understanding_runs=1,
            agent_max_context_runs=1,
            agent_max_response_runs=3,
            agent_max_safety_runs=5,
            agent_context_rag_timeout_seconds=0.01,
            agent_final_acceptance_min_confidence=0.6,
        )
        coordinator_agent = SimpleNamespace(
            name="CoordinatorAgent",
            root_task=lambda board: AgentTask(
                id="task:root",
                title="Resolve user turn",
                description=board.user_input,
                priority=TaskPriority.NORMAL,
                metadata={"kind": "root"},
            ),
            remember_acceptance=lambda artifact_id, reason: None,
        )
        registry = AgentRegistry([
            DemoAgent("AgentA", AgentCapability.UNDERSTANDING, 0.8),
            DemoAgent("AgentB", AgentCapability.SAFETY, 0.7),
        ])
        board = CollaborationBlackboard(turn_id="t1", user_input="hello", model_input="hello")

        result = EventDrivenCoordinator(registry, coordinator_agent, settings).run(board)

        claimed = [event.actor for event in result.events if event.type == AgentEventType.TASK_CLAIMED]
        self.assertIn("AgentA", claimed)
        self.assertIn("AgentB", claimed)


class WorkflowAgent:
    def __init__(self, name, capability, *, intent="CHAT", risk="LOW", delay=0.0, reject_candidates=False):
        self.profile = AgentProfile(
            name=name,
            capabilities=frozenset({capability}),
            system_prompt=f"{name} prompt",
            memory_policy="none",
            model_profile=name,
        )
        self.intent = intent
        self.risk = risk
        self.delay = delay
        self.reject_candidates = reject_candidates
        self.calls = 0

    def decide(self, task, board):
        return AgentDecision(True, 0.9, f"{self.profile.name} handles {task.id}")

    def act(self, task, board):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        name = self.profile.name
        if name == "UnderstandingAgent":
            artifact = AgentArtifact("intent-1", name, "intent", {"intent": self.intent}, task_id=task.id)
        elif name == "ContextAgent":
            artifact = AgentArtifact(
                "context-1", name, "context",
                {"memoryBrief": "brief", "modelHistory": [AiMessage(role="user", content=board.model_input)]},
                task_id=task.id,
            )
        elif name == "ResponseAgent":
            attempt = int(task.metadata.get("candidateAttempt", 1))
            artifact = AgentArtifact(
                f"response-{attempt}", name, "response_proposal",
                {"messages": [AiMessage(role="system", content="当前安全 可信任的人 心理中心 紧急服务")], "candidateAttempt": attempt},
                task_id=task.id, metadata={"candidateAttempt": attempt},
            )
        else:
            response = board.latest_artifact("response_proposal")
            if task.metadata.get("kind") == "risk" or response is None:
                artifact = AgentArtifact("risk-1", name, "risk", {"risk": self.risk}, task_id=task.id)
            else:
                fallback = bool(response.metadata.get("fallback"))
                approved = fallback or not self.reject_candidates
                artifact = AgentArtifact(
                    f"review-{self.calls}", name, "safety_review",
                    {
                        "approved": approved,
                        "reason": "missing targeted guidance" if not approved else "approved",
                        "responseArtifactId": response.id,
                    },
                    task_id=task.id,
                    metadata={"responseArtifactId": response.id},
                )
        return AgentTurnResult(artifacts=(artifact,))


class WorkflowCoordinatorTests(unittest.TestCase):
    def settings(self, timeout=0.01):
        return SimpleNamespace(
            agent_max_rounds=8,
            agent_max_understanding_runs=1,
            agent_max_context_runs=1,
            agent_max_response_runs=3,
            agent_max_safety_runs=5,
            agent_context_rag_timeout_seconds=timeout,
            agent_final_acceptance_min_confidence=0.6,
        )

    def coordinator_agent(self):
        return SimpleNamespace(
            name="CoordinatorAgent",
            root_task=lambda board: AgentTask(
                id="task:root", title="Resolve user turn", description=board.user_input,
                metadata={"kind": "root", "claimable": False},
            ),
            remember_acceptance=lambda artifact_id, reason: None,
        )

    def test_chat_low_skips_context_and_accepts_without_candidate_review(self):
        understanding = WorkflowAgent("UnderstandingAgent", AgentCapability.UNDERSTANDING, intent="CHAT")
        safety = WorkflowAgent("SafetyAgent", AgentCapability.SAFETY, risk="LOW")
        context = WorkflowAgent("ContextAgent", AgentCapability.CONTEXT)
        response = WorkflowAgent("ResponseAgent", AgentCapability.RESPONSE)
        board = CollaborationBlackboard(turn_id="low", user_input="hello", model_input="hello")

        result = EventDrivenCoordinator(
            AgentRegistry([understanding, safety, context, response]), self.coordinator_agent(), self.settings()
        ).run(board)

        self.assertEqual(result.workflow_status, "COMPLETED")
        self.assertEqual(context.calls, 0)
        self.assertEqual(safety.calls, 1)
        self.assertNotIn("task:gather-context", result.tasks)

    def test_non_high_response_waits_for_context_dependency(self):
        agents = [
            WorkflowAgent("UnderstandingAgent", AgentCapability.UNDERSTANDING, intent="CONSULT"),
            WorkflowAgent("SafetyAgent", AgentCapability.SAFETY, risk="LOW"),
            WorkflowAgent("ContextAgent", AgentCapability.CONTEXT),
            WorkflowAgent("ResponseAgent", AgentCapability.RESPONSE),
        ]
        result = EventDrivenCoordinator(
            AgentRegistry(agents), self.coordinator_agent(), self.settings()
        ).run(CollaborationBlackboard(turn_id="consult", user_input="压力大", model_input="压力大"))

        claimed = [(event.task_id, index) for index, event in enumerate(result.events) if event.type == AgentEventType.TASK_CLAIMED]
        context_index = next(index for task_id, index in claimed if task_id == "task:gather-context")
        response_index = next(index for task_id, index in claimed if task_id == "task:propose-response:1")
        self.assertLess(context_index, response_index)
        self.assertEqual(result.workflow_status, "COMPLETED")

    def test_high_response_uses_complete_context_before_response(self):
        context = WorkflowAgent("ContextAgent", AgentCapability.CONTEXT, delay=0.05)
        agents = [
            WorkflowAgent("UnderstandingAgent", AgentCapability.UNDERSTANDING, intent="RISK"),
            WorkflowAgent("SafetyAgent", AgentCapability.SAFETY, risk="HIGH"),
            context,
            WorkflowAgent("ResponseAgent", AgentCapability.RESPONSE),
        ]
        result = EventDrivenCoordinator(
            AgentRegistry(agents), self.coordinator_agent(), self.settings(timeout=0.001)
        ).run(CollaborationBlackboard(turn_id="high", user_input="不想活了", model_input="不想活了"))

        claimed = [(event.task_id, index) for index, event in enumerate(result.events) if event.type == AgentEventType.TASK_CLAIMED]
        context_index = next(index for task_id, index in claimed if task_id == "task:gather-context")
        response_index = next(index for task_id, index in claimed if task_id == "task:propose-response:1")
        self.assertLess(context_index, response_index)
        self.assertIsNotNone(result.latest_artifact("context"))
        self.assertEqual(result.workflow_status, "COMPLETED")

    def test_three_rejections_use_reviewed_fixed_fallback(self):
        safety = WorkflowAgent("SafetyAgent", AgentCapability.SAFETY, risk="HIGH", reject_candidates=True)
        response = WorkflowAgent("ResponseAgent", AgentCapability.RESPONSE)
        agents = [
            WorkflowAgent("UnderstandingAgent", AgentCapability.UNDERSTANDING, intent="RISK"),
            safety,
            WorkflowAgent("ContextAgent", AgentCapability.CONTEXT),
            response,
        ]
        result = EventDrivenCoordinator(
            AgentRegistry(agents), self.coordinator_agent(), self.settings(timeout=0.1)
        ).run(CollaborationBlackboard(turn_id="fallback", user_input="不想活了", model_input="不想活了"))

        latest = result.latest_artifact("response_proposal")
        self.assertEqual(response.calls, 3)
        self.assertEqual(safety.calls, 5)
        self.assertEqual(result.workflow_status, "FALLBACK_COMPLETED")
        self.assertTrue(latest.metadata.get("fallback"))
        self.assertEqual(len(result.artifacts_by_kind("response_proposal")), 1)
        self.assertFalse(any("revisionOf" in task.metadata for task in result.tasks.values()))


class ResponseRevisionTests(unittest.TestCase):
    def test_failed_internal_supplement_uses_program_guidance_block(self):
        class ExplodingClient:
            def complete(self, messages):
                raise RuntimeError("model unavailable")

        private_memory = SimpleNamespace(
            load=lambda agent_name, session_id: [],
            append=lambda agent_name, session_id, content: None,
            _key=lambda agent_name, session_id: f"agent:{agent_name}:{session_id}",
        )
        services = SimpleNamespace(
            model_registry=SimpleNamespace(client_for=lambda name: ExplodingClient()),
            private_memory=private_memory,
            session=SimpleNamespace(public_id="session-1"),
            user=SimpleNamespace(display_name="Student"),
        )
        agent = ResponseAgent(services)
        task = AgentTask(
            id="task:propose-response:2",
            title="Revise candidate",
            required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
            metadata={"candidateAttempt": 2, "lastFailureReason": "缺少可信任的人指导"},
        )
        board = CollaborationBlackboard(turn_id="revision", model_input="我不想活了", user_input="我不想活了")
        board = board.add_artifact(AgentArtifact("intent", "UnderstandingAgent", "intent", {"intent": "RISK"}))
        board = board.add_artifact(AgentArtifact("risk", "SafetyAgent", "risk", {"risk": "HIGH"}))

        result = agent.act(task, board)

        proposal = result.artifacts[0]
        combined = "\n".join(message.content for message in proposal.payload["messages"])
        self.assertTrue(proposal.payload["usedProgramFallback"])
        self.assertIn(HIGH_RISK_GUIDANCE_BLOCK, combined)


class ContextRagTimeoutTests(unittest.TestCase):
    def test_rag_timeout_keeps_memory_history_and_skills(self):
        class SlowRagContextAgent(ContextAgent):
            def _build_rag_context(self, memory_brief, model_input):
                time.sleep(0.05)
                return "late query", []

        history = [
            AiMessage(role="user", content="最近压力很大"),
            AiMessage(role="assistant", content="我们先慢慢梳理"),
        ]
        private_memory = SimpleNamespace(
            load=lambda agent_name, session_id: [],
            append=lambda agent_name, session_id, content: None,
            _key=lambda agent_name, session_id: f"agent:{agent_name}:{session_id}",
        )
        services = SimpleNamespace(
            settings=SimpleNamespace(
                agent_context_rag_timeout_seconds=0.001,
                agent_context_summary_timeout_seconds=0.1,
                memory_summary_max_chars=500,
                chat_history_limit=10,
                knowledge_top_k=4,
            ),
            model_registry=SimpleNamespace(client_for=lambda name: SimpleNamespace(complete=lambda messages: "历史摘要")),
            private_memory=private_memory,
            memory=SimpleNamespace(load_recent=lambda session_id: list(history)),
            session=SimpleNamespace(public_id="session-1", id=-1),
            user=SimpleNamespace(display_name="Student"),
            db=SimpleNamespace(),
        )
        agent = SlowRagContextAgent(services)
        task = AgentTask(id="task:gather-context", title="Context", metadata={"kind": "context"})
        board = CollaborationBlackboard(turn_id="rag-timeout", user_input="压力很大", model_input="压力很大")
        board = board.add_artifact(AgentArtifact("intent", "UnderstandingAgent", "intent", {"intent": "CONSULT"}))
        board = board.add_artifact(AgentArtifact("risk", "SafetyAgent", "risk", {"risk": "HIGH"}))

        result = agent.act(task, board)

        context = result.artifacts[0].payload
        self.assertEqual(context["memoryBrief"], "历史摘要")
        self.assertTrue(context["modelHistory"])
        self.assertTrue(context["skillContext"])
        self.assertEqual(context["retrievedKnowledge"], [])
        self.assertTrue(context["ragTimedOut"])
        self.assertEqual(result.events[0].metadata["stage"], "RAG")


class AgentModelRegistryTests(unittest.TestCase):
    def test_agent_can_override_model_without_changing_global_default(self):
        settings = SimpleNamespace(
            ai_provider="ollama",
            ollama_model="default-model",
            openai_model="default-openai",
            ai_temperature=0.35,
            ai_max_tokens=512,
            agent_model_understanding_model="small-understanding-model",
            agent_model_understanding_provider="ollama",
            agent_model_safety_model="safety-model",
            agent_model_safety_provider="ollama",
        )

        registry = AgentModelRegistry(settings)

        self.assertEqual(registry.profile_for("UnderstandingAgent").model, "small-understanding-model")
        self.assertEqual(registry.profile_for("ResponseAgent").model, "default-model")


if __name__ == "__main__":
    unittest.main()
