
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from loguru import logger



# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# SRC_ROOT = PROJECT_ROOT / "src"
# if str(SRC_ROOT) not in sys.path:
#     sys.path.insert(0, str(SRC_ROOT))



logger.disable("tau2")

from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType  # noqa: E402
from tau2.run import get_tasks, run_task  # noqa: E402


DEFAULT_TRIAL_SEEDS = (300, 301)
NUM_TRIALS = 2
DEFAULT_AGENT_REQUEST_TIMEOUT = 900.0
DEFAULT_USER_REQUEST_TIMEOUT = 300.0
DEFAULT_AGENT_REQUEST_RETRIES = 1
DEFAULT_USER_REQUEST_RETRIES = 2
DEFAULT_AGENT_REQUEST_RETRY_SLEEP = 2.0
DEFAULT_USER_REQUEST_RETRY_SLEEP = 2.0
AGENT_API_BASE = os.getenv("TAU2_AGENT_API_BASE", os.getenv("WER_EXECUTOR_BASE_URL", "http://localhost:8000/v1"))
AGENT_MODEL_NAME = os.getenv("TAU2_AGENT_MODEL", f"openai/{os.getenv('WER_EXECUTOR_MODEL', 'executor-model')}")
USER_MODEL_NAME = os.getenv("TAU2_USER_MODEL", "openai/user-simulator-model")

class Tau2InfrastructureError(RuntimeError):
    """A trial could not be evaluated because its execution infrastructure failed."""


@dataclass(frozen=True)
class Tau2TrialResult:
    """Minimal reward and task-completion trajectory for one Tau2 trial."""

    reward: float
    trajectory: list[dict[str, Any]]


@dataclass(frozen=True)
class Tau2TaskEvaluationResult:
    """Aggregated result returned to verl and diagnostic consumers."""

    domain: str
    task_id: str
    acc_reward: float
    trials: list[Tau2TrialResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete result to a JSON-serializable dictionary."""

        return asdict(self)


def _default_agent_llm_args() -> dict[str, Any]:
    args: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 4096,
        "timeout": DEFAULT_AGENT_REQUEST_TIMEOUT,
        "num_retries": DEFAULT_AGENT_REQUEST_RETRIES,
        "tau2_retry_sleep": DEFAULT_AGENT_REQUEST_RETRY_SLEEP,
    }
    api_base = AGENT_API_BASE
    if api_base:
        args["api_base"] = api_base
    api_key = os.getenv("TAU2_AGENT_API_KEY", os.getenv("WER_EXECUTOR_API_KEY", "EMPTY"))
    if api_key:
        args["api_key"] = api_key
    return args


def _default_user_llm_args() -> dict[str, Any]:
    args: dict[str, Any] = {
        "temperature": 0.0,
        "timeout": float(os.getenv("TAU2_USER_REQUEST_TIMEOUT", str(DEFAULT_USER_REQUEST_TIMEOUT))),
        "num_retries": int(os.getenv("TAU2_USER_REQUEST_RETRIES", str(DEFAULT_USER_REQUEST_RETRIES))),
        "tau2_retry_sleep": float(os.getenv("TAU2_USER_REQUEST_RETRY_SLEEP", str(DEFAULT_USER_REQUEST_RETRY_SLEEP))),
    }
    api_base = os.getenv("TAU2_USER_API_BASE") or os.getenv("DEEPSEEK_API_BASE")
    api_key = os.getenv("TAU2_USER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if api_base:
        args["api_base"] = api_base
    if api_key:
        args["api_key"] = api_key
    return args


def _serialize_trajectory(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Keep only dialogue text, tool calls, tool arguments and tool outputs."""

    trajectory: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}

    def append_tool_result(message: ToolMessage) -> None:
        event: dict[str, Any] = {
            "type": "tool_result",
            "content": message.content,
        }
        tool_name = tool_names_by_id.get(message.id)
        if tool_name is not None:
            event["name"] = tool_name
        if message.error:
            event["error"] = True
        trajectory.append(event)

    for message in messages:
        if isinstance(message, SystemMessage):
            continue

        if isinstance(message, (UserMessage, AssistantMessage)):
            if message.content is not None and message.content.strip():
                trajectory.append(
                    {
                        "type": message.role,
                        "content": message.content,
                    }
                )
            for tool_call in message.tool_calls or []:
                if tool_call.id:
                    tool_names_by_id[tool_call.id] = tool_call.name
                trajectory.append(
                    {
                        "type": "tool_call",
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                )
            continue

        if isinstance(message, ToolMessage):
            append_tool_result(message)
            continue

        if isinstance(message, MultiToolMessage):
            for tool_message in message.tool_messages:
                append_tool_result(tool_message)

    return trajectory


def _serialize_trial(
    simulation: SimulationRun,
    *,
    trial_index: int,
    seed: int,
) -> Tau2TrialResult:
    if simulation.reward_info is None:
        raise Tau2InfrastructureError(
            f"Trial {trial_index} (seed={seed}) finished without reward_info"
        )

    return Tau2TrialResult(
        reward=float(simulation.reward_info.reward),
        trajectory=_serialize_trajectory(simulation.messages),
    )


def evaluate_tau2_task(
    domain: str = "airline",
    task_id: str = "0",
    *,
    task_split_name: str = "base",
    agent: str = "llm_agent",
    agent_llm: str = AGENT_MODEL_NAME,
    agent_llm_args: Mapping[str, Any] | None = None,
    user: str = "user_simulator",
    user_llm: str = USER_MODEL_NAME,
    user_llm_args: Mapping[str, Any] | None = None,
    trial_seeds: Sequence[int] = DEFAULT_TRIAL_SEEDS,
    max_concurrency: int = 2,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    enforce_communication_protocol: bool = False,
    skill: str | None = None,
) -> Tau2TaskEvaluationResult:
    """Evaluate exactly one Tau2 task using exactly two fixed trial seeds.

    Business failures represented by Tau2 as unsuccessful trajectories are returned
    normally with reward 0. Exceptions raised while invoking a model or constructing
    the simulation are surfaced as ``Tau2InfrastructureError`` so the caller can retry
    instead of treating an infrastructure problem as a bad skill.
    """

    if len(trial_seeds) != NUM_TRIALS:
        raise ValueError("trial_seeds must contain exactly two seeds")

    normalized_seeds = tuple(int(seed) for seed in trial_seeds)
    if normalized_seeds[0] == normalized_seeds[1]:
        raise ValueError("the two trial seeds must be different")
    if not 1 <= max_concurrency <= NUM_TRIALS:
        raise ValueError("max_concurrency must be 1 or 2")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("max_errors must be greater than 0")

    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split_name,
        task_ids=[str(task_id)],
    )
    if len(tasks) != 1:
        raise ValueError(
            f"Expected exactly one task for {domain}/{task_split_name}/{task_id}, "
            f"found {len(tasks)}"
        )
    task = tasks[0]

    base_agent_args = (
        _default_agent_llm_args()
        if agent_llm_args is None
        else dict(agent_llm_args)
    )
    base_user_args = (
        _default_user_llm_args()
        if user_llm_args is None
        else dict(user_llm_args)
    )

    evaluation_id = uuid4().hex[:8]
    normalized_skill = skill.strip() if skill and skill.strip() else None
    skill_id = (
        hashlib.sha256(normalized_skill.encode("utf-8")).hexdigest()[:8]
        if normalized_skill is not None
        else "none"
    )
    started_at = time.perf_counter()
    print(
        f"[tau2][{evaluation_id}] START domain={domain} task={task_id} "
        f"skill={skill_id} trials={NUM_TRIALS}",
        flush=True,
    )

    def execute_trial(trial_index: int, seed: int) -> Tau2TrialResult:
        def report_response_completed(role: str, step: int) -> None:
            print(
                f"[tau2][{evaluation_id}][trial={trial_index} seed={seed}]"
                f"[step={step}] {role} response completed",
                flush=True,
            )

        def report_request_retry(role: str):
            def callback(
                retry_number: int,
                max_retries: int,
                retry_sleep: float,
                error: Exception,
            ) -> None:
                print(
                    f"[tau2][{evaluation_id}][trial={trial_index} seed={seed}] "
                    f"{role} request retry {retry_number}/{max_retries} "
                    f"in {retry_sleep:g}s after {type(error).__name__}",
                    flush=True,
                )

            return callback

        try:
            
            trial_agent_args = copy.deepcopy(base_agent_args)
            trial_user_args = copy.deepcopy(base_user_args)
            trial_agent_args["tau2_retry_callback"] = report_request_retry("AGENT")
            trial_user_args["tau2_retry_callback"] = report_request_retry("USER")
            simulation = run_task(
                domain=domain,
                task=task,
                agent=agent,
                user=user,
                llm_agent=agent_llm,
                llm_args_agent=trial_agent_args,
                llm_user=user_llm,
                llm_args_user=trial_user_args,
                max_steps=max_steps,
                max_errors=max_errors,
                evaluation_type=evaluation_type,
                seed=seed,
                enforce_communication_protocol=enforce_communication_protocol,
                skill=skill,
                progress_callback=report_response_completed,
            )
            simulation.trial = trial_index
            return _serialize_trial(
                simulation,
                trial_index=trial_index,
                seed=seed,
            )
        except Tau2InfrastructureError:
            raise
        except Exception as exc:
            raise Tau2InfrastructureError(
                f"Tau2 trial {trial_index} (seed={seed}) failed: {exc}"
            ) from exc

    try:
        completed_trials: list[Tau2TrialResult | None] = [None] * NUM_TRIALS
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(execute_trial, index, seed): index
                for index, seed in enumerate(normalized_seeds)
            }
            for future in as_completed(futures):
                trial_index = futures[future]
                completed_trials[trial_index] = future.result()

        if any(trial is None for trial in completed_trials):
            raise Tau2InfrastructureError("Not all Tau2 trials produced a result")
        trials = [trial for trial in completed_trials if trial is not None]

        reward_sum = sum(trial.reward for trial in trials)
        result = Tau2TaskEvaluationResult(
            domain=domain,
            task_id=str(task_id),
            acc_reward=reward_sum,
            trials=trials,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(
            f"[tau2][{evaluation_id}] ERROR domain={domain} task={task_id} "
            f"elapsed={elapsed:.1f}s {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    elapsed = time.perf_counter() - started_at
    trial_rewards = [trial.reward for trial in result.trials]
    print(
        f"[tau2][{evaluation_id}] DONE domain={domain} task={task_id} "
        f"reward={result.acc_reward} trials={trial_rewards} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


