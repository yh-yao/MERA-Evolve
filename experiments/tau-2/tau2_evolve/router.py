"""Cost-aware turn router and deterministic training features for tau2."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FEATURE_NAMES = (
    "step_fraction",
    "assistant_turn_fraction",
    "tool_call_fraction",
    "tool_error_fraction",
    "unique_tool_fraction",
    "repeated_tool_fraction",
    "repeated_last_tool",
    "last_assistant_without_tool",
    "last_input_is_tool_error",
    "context_size",
)


def _as_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return dict(message)


def _tool_names(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = call.get("name") or function.get("name")
            if name:
                names.append(str(name))
    return names


def router_features(messages: list[Any], max_steps: int = 40) -> dict[str, float]:
    rows = [_as_dict(message) for message in messages]
    assistant = [row for row in rows if row.get("role") == "assistant"]
    tools = _tool_names(rows)
    tool_messages = [row for row in rows if row.get("role") == "tool"]
    tool_errors = sum(
        bool(row.get("error")) or str(row.get("content") or "").startswith("Error:")
        for row in tool_messages
    )
    unique_tools = len(set(tools))
    repeated_tools = len(tools) - unique_tools
    last_assistant = assistant[-1] if assistant else {}
    last_row = rows[-1] if rows else {}
    context_chars = sum(len(str(row.get("content") or "")) for row in rows)
    return {
        "step_fraction": min(len(rows) / max(max_steps, 1), 2.0),
        "assistant_turn_fraction": min(len(assistant) / max(max_steps / 2, 1), 2.0),
        "tool_call_fraction": min(len(tools) / max(max_steps / 2, 1), 2.0),
        "tool_error_fraction": tool_errors / max(len(tool_messages), 1),
        "unique_tool_fraction": min(unique_tools / max(max_steps / 2, 1), 2.0),
        "repeated_tool_fraction": repeated_tools / max(len(tools), 1),
        "repeated_last_tool": float(len(tools) >= 2 and tools[-1] == tools[-2]),
        "last_assistant_without_tool": float(bool(last_assistant) and not last_assistant.get("tool_calls")),
        "last_input_is_tool_error": float(
            last_row.get("role") == "tool"
            and (bool(last_row.get("error")) or str(last_row.get("content") or "").startswith("Error:"))
        ),
        "context_size": min(math.log1p(context_chars) / 12.0, 1.5),
    }


@dataclass
class RouterModel:
    weights: list[float]
    bias: float
    threshold: float = 0.5

    @classmethod
    def load(cls, path: str | Path) -> "RouterModel":
        payload = json.loads(Path(path).read_text())
        if payload["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("router feature schema mismatch")
        return cls(payload["weights"], payload["bias"], payload.get("threshold", 0.5))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
        }, indent=2) + "\n")

    def probability(self, features: dict[str, float]) -> float:
        score = self.bias + sum(
            weight * float(features.get(name, 0.0))
            for name, weight in zip(FEATURE_NAMES, self.weights)
        )
        score = max(min(score, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-score))

    def should_escalate(self, features: dict[str, float]) -> bool:
        return self.probability(features) >= self.threshold

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, Any]],
        *,
        learning_rate: float = 0.1,
        epochs: int = 500,
        l2: float = 0.01,
        threshold: float = 0.5,
    ) -> "RouterModel":
        if not rows or len({int(row["label"]) for row in rows}) < 2:
            raise ValueError("router training requires both continue and escalate examples")
        weights = [0.0] * len(FEATURE_NAMES)
        positives = sum(int(row["label"]) for row in rows)
        bias = math.log((positives + 1.0) / (len(rows) - positives + 1.0))
        for _ in range(epochs):
            grad_w = [0.0] * len(weights)
            grad_b = 0.0
            for row in rows:
                vector = [float(row["features"].get(name, 0.0)) for name in FEATURE_NAMES]
                score = max(min(bias + sum(w * x for w, x in zip(weights, vector)), 30.0), -30.0)
                error = 1.0 / (1.0 + math.exp(-score)) - int(row["label"])
                grad_b += error
                for index, value in enumerate(vector):
                    grad_w[index] += error * value
            scale = 1.0 / len(rows)
            bias -= learning_rate * grad_b * scale
            for index in range(len(weights)):
                weights[index] -= learning_rate * (grad_w[index] * scale + l2 * weights[index])
        return cls(weights, bias, threshold)


@dataclass
class RoutedAgentState:
    active_state: Any
    escalated: bool = False


class RoutedAgent:
    """Switch from a student agent to a teacher at an agent decision point."""

    def __init__(self, student, teacher, model: RouterModel, max_steps: int):
        self.student = student
        self.teacher = teacher
        self.model = model
        self.max_steps = max_steps
        self.escalations = 0
        self.route_probabilities: list[float] = []

    def __getattr__(self, name: str):
        return getattr(self.student, name)

    def set_seed(self, seed: int) -> None:
        self.student.set_seed(seed)
        self.teacher.set_seed(seed)

    def get_init_state(self, message_history=None) -> RoutedAgentState:
        return RoutedAgentState(self.student.get_init_state(message_history=message_history))

    def generate_next_message(self, message, state: RoutedAgentState):
        history = list(getattr(state.active_state, "messages", [])) + [message]
        features = router_features(history, self.max_steps)
        probability = self.model.probability(features)
        self.route_probabilities.append(probability)
        if not state.escalated and probability >= self.model.threshold:
            prior = list(getattr(state.active_state, "messages", []))
            state.active_state = self.teacher.get_init_state(message_history=prior)
            state.escalated = True
            self.escalations += 1
        agent = self.teacher if state.escalated else self.student
        response, state.active_state = agent.generate_next_message(message, state.active_state)
        return response, state

    def is_stop(self, message) -> bool:
        return self.student.is_stop(message)

    def stop(self, message, state: RoutedAgentState | None):
        if state is None:
            return self.student.stop(message, None)
        agent = self.teacher if state.escalated else self.student
        return agent.stop(message, state.active_state)
