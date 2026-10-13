"""
Preprocess a custom JSONL dataset to parquet format for verl.
"""

import os
import json
from functools import lru_cache
from pathlib import Path

import datasets
from transformers import AutoTokenizer

try:
    from .extract_trajectory import remove_system_messages
except ImportError:
    from extract_trajectory import remove_system_messages


# Fill these paths before running.
DATASET_PATH = os.getenv("BFCL_INITIAL_CONTEXT_QUERY_PATH", str(Path(__file__).resolve().parents[2] / "berkeley-function-call-leaderboard/bfcl_eval/data/compiled/BFCL_v4_multi_turn_base_initial_context_query.jsonl"))
LOCAL_SAVE_DIR = os.getenv("WER_DATA_DIR", str(Path(__file__).resolve().parents[3] / "extra_datasets"))
DATA_SOURCE = "custom/skill4bfcl"
ABILITY = "skill optimization"
TOKENIZER_MODEL_PATH = os.getenv("TOKENIZER_MODEL_PATH") or os.getenv("BASE_MODEL", "/path/to/models/skill-optimizer")
DROP_PROMPT_TOKEN_THRESHOLD = 18000
TARGET_TRAIN_REWARD = 1.0
REWARD_TOLERANCE = 0.5

DEFAULT_SKILL_INSTANCE = """
# Name: ...(a concise skill name that describes the reusable task pattern)

## Description
...(a one-sentence summary of the task family and the agent behavior this skill should guide)

## Workflow
1. ...
2. ...
3. ...
...
n. ...

## Notes
- ...(important assumptions, limitations, or edge cases to keep in mind)
- ...(common mistakes or failure modes the agent should avoid)
"""


@lru_cache(maxsize=1)
def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def count_tokens(text, model_path=TOKENIZER_MODEL_PATH):
    tokenizer = load_tokenizer(model_path)
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_example_reward(example):
    reward = example.get("reward")
    if reward is None:
        return None

    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def should_drop_example(prompt_token_count):
    return prompt_token_count > DROP_PROMPT_TOKEN_THRESHOLD


def should_keep_example_reward(example_reward):
    return (
        example_reward is not None
        and abs(example_reward - TARGET_TRAIN_REWARD) < REWARD_TOLERANCE
    )


def _format_context(value):
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def split_trajectories_by_reward(trajs):
    success_trajectories = []
    failed_trajectories = []

    if not isinstance(trajs, list):
        return success_trajectories, failed_trajectories

    for traj_item in trajs:
        if not isinstance(traj_item, dict):
            continue

        try:
            reward = float(traj_item.get("reward", 0))
        except (TypeError, ValueError):
            continue

        trajectory = traj_item.get("trajectory", [])
        trajectory_without_system = remove_system_messages(trajectory)
        if reward > 0.5:
            success_trajectories.append(trajectory_without_system)
        elif reward < 0.5:
            failed_trajectories.append(trajectory_without_system)

    return success_trajectories, failed_trajectories


def build_prompt(item, system_content, instance=None):
    current_skill = _format_context(item.get("skill") or item.get("sequence_1", ""))
    trajs = item.get("trajectories", [])
    success_trajs, failed_trajs = split_trajectories_by_reward(trajs)
    system_content = _format_context(system_content)
    success_trajs = _format_context(success_trajs)
    failed_trajs = _format_context(failed_trajs)

    skill_instance = _format_context(
        DEFAULT_SKILL_INSTANCE if instance is None else instance
    )
    return f"""A skill is a reusable prompt that teaches a model how to handle a class of similar tasks.

The following is only a skill format example. Its content is unrelated to the current task; only follow its format.

{skill_instance}

Based on the current skill, system content, successful trajectories, and failed trajectories below, optimize the current skill into a more reusable skill for future similar tasks.

<current_skill>
{current_skill}
</current_skill>

<system_content>
{system_content}
</system_content>

<success_trajectories>
{success_trajs}
</success_trajectories>

<failed_trajectories>
{failed_trajs}
</failed_trajectories>

Optimization guidelines:
- Keep the same markdown structure as the format example.
- Preserve the reusable task pattern in the current skill, but refine it using evidence from successful and failed trajectories.
- refine the workflow-style guidance that helps an agent understand the task, choose actions, and produce the final response.
- Use successful trajectories to reinforce effective high-level behaviors.
- Use failed trajectories to summarize cautions, edge cases, and common failure modes in the Notes section.
- Keep the workflow high-level and transferable. Do not write a concrete function-call trace, exact argument values, specific IDs, or parameter-by-parameter instructions.
- Do not overfit to a single trajectory, user query, entity name, or tool output unless it represents a general reusable pattern.
- Do not mention trajectory IDs, reward values, or internal preprocessing details.
- Do not answer the original query.

Return only the optimized skill wrapped in XML tags exactly as follows:
<skill>
...optimized skill...
</skill>
"""


def make_map_fn(split, initial_context_by_run_id):
    def process_fn(example, idx):
        return build_verl_record(example, idx, split, initial_context_by_run_id)

    return process_fn


def iter_jsonl_records(dataset_path):
    with open(dataset_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {dataset_path} at line {line_no}: {e}") from e

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {dataset_path} at line {line_no}, got {type(record)}")

            yield record


def _get_run_id(example):
    run_id = example.get("run_id")
    if run_id is None:
        run_id = example.get("runid")
    return run_id


def load_initial_context_by_run_id(initial_context_path):
    initial_context_path = os.path.expanduser(str(initial_context_path))
    initial_context_by_run_id = {}

    for idx, item in enumerate(iter_jsonl_records(initial_context_path)):
        run_id = _get_run_id(item)
        if run_id is None or run_id == "":
            raise ValueError(
                f"Missing run_id/runid in initial context file index {idx}"
            )

        initial_context = item.get("initial_context")
        if initial_context is None or initial_context == "":
            raise ValueError(f"Missing initial_context for run_id {run_id}")

        run_id = str(run_id)
        if run_id in initial_context_by_run_id:
            raise ValueError(f"Duplicate run_id in initial context file: {run_id}")

        initial_context_by_run_id[run_id] = initial_context

    if not initial_context_by_run_id:
        raise ValueError(f"No initial context records found in {initial_context_path}")

    return initial_context_by_run_id


def build_verl_record(example, idx, split, initial_context_by_run_id):
    run_id = _get_run_id(example)
    if run_id is None or run_id == "":
        raise ValueError(f"Missing run_id/runid in record index {idx}")

    system_content = initial_context_by_run_id.get(str(run_id))
    if system_content is None:
        raise ValueError(f"Missing initial_context for run_id {run_id}")

    sequence_1 = example.get("sequence_1")
    if sequence_1 is None or sequence_1 == "":
        raise ValueError(f"Missing sequence_1 in record index {idx}")

    example_reward = extract_example_reward(example)
    # if not should_keep_example_reward(example_reward):
    #     print(f"skip non-1 reward sample: reward={example_reward}, run_id={run_id}")
    #     return None

    prompt_content = build_prompt(item=example, system_content=system_content)
    prompt_text = (
        prompt_content
        if isinstance(prompt_content, str)
        else _format_context(prompt_content)
    )
    prompt_token_count = count_tokens(prompt_text)

    if should_drop_example(prompt_token_count):
        print(f"has dropped a {prompt_token_count} token sample")
        return None

    prompt = [{"role": "user", "content": prompt_text}]

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": "default"},
        "extra_info": {
            "split": split,
            "index": idx,
            "run_id": run_id,
            "pre_skill": sequence_1,
            "prompt_token_count": prompt_token_count,
            "example_reward": example_reward,
        },
    }


def main(dataset_path, parquet_output_path):
    
    
    from verl.datapreprocess import build_seed_skill_train_parquet as mixed_builder

    dataset_path = os.path.expanduser(str(dataset_path))
    parquet_output_path = os.path.expanduser(str(parquet_output_path))

    raw_records = list(iter_jsonl_records(dataset_path))
    if not raw_records:
        raise ValueError(f"No failed skill records found in {dataset_path}")

    
    
    routed_rollouts = [
        mixed_builder.RoutedRollout(
            dataset=mixed_builder.detect_dataset(record, source_index),
            source_index=source_index,
            record=record,
        )
        for source_index, record in enumerate(raw_records)
    ]
    input_counts = mixed_builder.count_by_dataset(routed_rollouts)

    bfcl_initial_contexts = None
    if input_counts[mixed_builder.DATASET_BFCL]:
        bfcl_initial_contexts = load_initial_context_by_run_id(
            os.getenv("BFCL_INITIAL_CONTEXT_QUERY_PATH", str(mixed_builder.DEFAULT_INITIAL_CONTEXT_PATH))
        )

    tau2_initial_contexts = None
    if input_counts[mixed_builder.DATASET_TAU2]:
        tau2_initial_contexts = mixed_builder.load_tau2_initial_context_by_task(
            os.getenv("TAU2_INITIAL_CONTEXT_QUERY_PATH", str(mixed_builder.DEFAULT_TAU2_INITIAL_CONTEXT_PATH))
        )

    built_records, skipped_by_dataset = mixed_builder.build_routed_records(
        routed_rollouts,
        split="train",
        bfcl_initial_contexts=bfcl_initial_contexts,
        tau2_initial_contexts=tau2_initial_contexts,
    )
    if not built_records:
        raise ValueError(f"No train records were built from {dataset_path}")

    
    
    records = [item.record for item in built_records]
    train_dataset = datasets.Dataset.from_list(
        records,
        features=mixed_builder.build_train_features(),
    )

    output_dir = os.path.dirname(parquet_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    train_dataset.to_parquet(parquet_output_path)
    print(f"Train parquet saved to {parquet_output_path}")
    print(f"Total records: {len(routed_rollouts)}")
    print(f"Input records by dataset: {input_counts}")
    print(f"Kept records: {len(records)}")
    print(
        "Skipped records during build_verl_record: "
        f"{sum(skipped_by_dataset.values())} ({skipped_by_dataset})"
    )
    return parquet_output_path


if __name__ == "__main__":
    main(DATASET_PATH, os.path.join(LOCAL_SAVE_DIR, "train_small.parquet"))
