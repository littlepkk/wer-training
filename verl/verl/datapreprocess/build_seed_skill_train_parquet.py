
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import datasets

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VERL_ROOT = WORKSPACE_ROOT / "verl"
BFCL_ROOT = VERL_ROOT / "berkeley-function-call-leaderboard"
TAU2_ROOT = VERL_ROOT / "tau2-bench-verified"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.datapreprocess import skill4bfcl4phase as phase  # noqa: E402

DEFAULT_ORIGIN_DIR = Path(os.getenv("WER_DATA_DIR", str(WORKSPACE_ROOT / "extra_datasets"))) / "origin"
CATEGORY_KEEP_LIMIT = 13
RUN_ID_PREFIX = "multi_turn_base_"
DATASET_BFCL = "bfcl"
DATASET_TAU2 = "tau2"
TAU2_DOMAINS = frozenset({"airline", "retail", "telecom"})
TASK_CLASSES = [
    {
        "key": "filesystem",
        "range": "0-49",
        "start": 0,
        "end": 49,
        "description": "文件系统类任务",
        "main_class": "GorillaFileSystem",
    },
    {
        "key": "vehicle_control",
        "range": "50-99",
        "start": 50,
        "end": 99,
        "description": "车控类任务",
        "main_class": "VehicleControlAPI",
    },
    {
        "key": "trading",
        "range": "100-149",
        "start": 100,
        "end": 149,
        "description": "股票交易类任务",
        "main_class": "TradingBot",
    },
    {
        "key": "travel",
        "range": "150-199",
        "start": 150,
        "end": 199,
        "description": "旅行订票类任务",
        "main_class": "TravelAPI",
    },
]
DEFAULT_INITIAL_CONTEXT_PATH = (
    BFCL_ROOT / "bfcl_eval" / "data" / "compiled" / "BFCL_v4_multi_turn_base_initial_context_query.jsonl"
)
DEFAULT_TAU2_INITIAL_CONTEXT_PATH = TAU2_ROOT / "data" / "tau2" / "compiled" / "tau2_train_initial_context_query.jsonl"


@dataclass(frozen=True)
class RoutedRollout:
    dataset: Literal["bfcl", "tau2"]
    source_index: int
    record: dict[str, Any]


@dataclass(frozen=True)
class BuiltRecord:
    dataset: Literal["bfcl", "tau2"]
    source_index: int
    record: dict[str, Any]


def maybe_patch_token_counter(phase_module, tokenizer_model_path: str | None) -> None:
    if not tokenizer_model_path:
        return

    original_count_tokens = phase_module.count_tokens

    
    
    def count_tokens_with_override(text: str) -> int:
        return original_count_tokens(text, model_path=tokenizer_model_path)

    phase_module.count_tokens = count_tokens_with_override


def _normalize_id(value: Any, *, field_name: str, record_index: int) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):  # noqa: UP038
        raise ValueError(f"Invalid {field_name} in rollout record index {record_index}: {value!r}")
    normalized = str(value)
    if not normalized:
        raise ValueError(f"Empty {field_name} in rollout record index {record_index}")
    return normalized


def detect_dataset(
    record: dict[str, Any],
    record_index: int,
) -> Literal["bfcl", "tau2"]:
    has_domain = "domain" in record
    if not has_domain:
        return DATASET_BFCL

    domain = record.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"Invalid Tau2 domain in rollout record index {record_index}: {domain!r}")
    if domain not in TAU2_DOMAINS:
        raise ValueError(f"Unsupported Tau2 domain in rollout record index {record_index}: {domain!r}")
    return DATASET_TAU2


def route_rollouts(records: list[dict[str, Any]]) -> list[RoutedRollout]:
    routed: list[RoutedRollout] = []
    seen_keys: set[tuple[str, ...]] = set()

    for source_index, record in enumerate(records):
        dataset = detect_dataset(record, source_index)
        raw_run_id = record.get("run_id")
        if raw_run_id is None:
            raw_run_id = record.get("runid")
        run_id = _normalize_id(
            raw_run_id,
            field_name="run_id/runid",
            record_index=source_index,
        )

        if dataset == DATASET_TAU2:
            domain = str(record["domain"])
            unique_key = (dataset, domain, run_id)
        else:
            unique_key = (dataset, run_id)
        if unique_key in seen_keys:
            raise ValueError(f"Duplicate {dataset} rollout task key at record index {source_index}: {unique_key!r}")
        seen_keys.add(unique_key)
        routed.append(RoutedRollout(dataset, source_index, record))

    return routed


def load_tau2_initial_context_by_task(
    initial_context_path: Path,
) -> dict[tuple[str, str], Any]:
    initial_context_by_task: dict[tuple[str, str], Any] = {}

    for idx, item in enumerate(phase.iter_jsonl_records(initial_context_path)):
        domain = item.get("domain")
        if not isinstance(domain, str) or domain not in TAU2_DOMAINS:
            raise ValueError(f"Invalid Tau2 domain in initial context index {idx}: {domain!r}")

        raw_run_id = item.get("run_id")
        if raw_run_id is None:
            raw_run_id = item.get("runid")
        run_id = _normalize_id(
            raw_run_id,
            field_name="run_id/runid",
            record_index=idx,
        )

        initial_context = item.get("initial_context")
        if initial_context is None or initial_context == "":
            raise ValueError(f"Missing initial_context for Tau2 task {domain}/{run_id}")

        key = (domain, run_id)
        if key in initial_context_by_task:
            raise ValueError(f"Duplicate Tau2 initial context task key: {key!r}")
        initial_context_by_task[key] = initial_context

    if not initial_context_by_task:
        raise ValueError(f"No Tau2 initial context records found in {initial_context_path}")
    return initial_context_by_task


def validate_tau2_rollout(example: dict[str, Any], idx: int) -> tuple[str, str]:
    domain = example.get("domain")
    if not isinstance(domain, str) or domain not in TAU2_DOMAINS:
        raise ValueError(f"Invalid Tau2 domain in record index {idx}: {domain!r}")

    raw_run_id = example.get("run_id")
    if raw_run_id is None:
        raw_run_id = example.get("runid")
    run_id = _normalize_id(
        raw_run_id,
        field_name="run_id/runid",
        record_index=idx,
    )

    skill = example.get("skill")
    if not isinstance(skill, str) or not skill.strip():
        raise ValueError(f"Missing or empty skill in Tau2 record index {idx}")

    reward = example.get("reward")
    try:
        float(reward)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid reward in Tau2 record index {idx}: {reward!r}") from error

    trajectories = example.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 2:
        raise ValueError(f"Tau2 record index {idx} must contain exactly two trajectories")
    for trajectory_index, trajectory_item in enumerate(trajectories):
        if not isinstance(trajectory_item, dict):
            raise ValueError(f"Invalid Tau2 trajectory {trajectory_index} in record index {idx}")
        trial_reward = trajectory_item.get("reward")
        try:
            float(trial_reward)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid Tau2 trajectory reward at record index {idx}, trajectory {trajectory_index}: {trial_reward!r}"
            ) from error
        if not isinstance(trajectory_item.get("trajectory"), list):
            raise ValueError(f"Invalid Tau2 trajectory payload at record index {idx}, trajectory {trajectory_index}")

    return domain, run_id


def build_tau2_verl_record(
    example: dict[str, Any],
    idx: int,
    split: str,
    initial_context_by_task: dict[tuple[str, str], Any],
) -> dict[str, Any] | None:
    domain, run_id = validate_tau2_rollout(example, idx)
    system_content = initial_context_by_task.get((domain, run_id))
    if system_content is None:
        raise ValueError(f"Missing initial_context for Tau2 task {domain}/{run_id}")

    prompt_content = phase.build_prompt(item=example, system_content=system_content)
    prompt_text = prompt_content if isinstance(prompt_content, str) else phase._format_context(prompt_content)
    prompt_token_count = phase.count_tokens(prompt_text)
    if phase.should_drop_example(prompt_token_count):
        print(f"has dropped a {prompt_token_count} token Tau2 sample: domain={domain}, run_id={run_id}")
        return None

    return {
        "data_source": phase.DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": phase.ABILITY,
        "reward_model": {"style": "rule", "ground_truth": "default"},
        "extra_info": {
            "split": split,
            "index": idx,
            "run_id": run_id,
            "pre_skill": None,
            "prompt_token_count": prompt_token_count,
            "example_reward": phase.extract_example_reward(example),
        },
    }


def normalize_verl_record_schema(
    record: dict[str, Any],
    *,
    dataset: Literal["bfcl", "tau2"],
    domain: str | None,
) -> dict[str, Any]:
    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError("Built verl record is missing extra_info")

    run_id = extra_info.get("run_id")
    if run_id is None or run_id == "":
        raise ValueError("Built verl record is missing extra_info.run_id")
    pre_skill = extra_info.get("pre_skill")
    example_reward = extra_info.get("example_reward")

    return {
        "data_source": record["data_source"],
        "prompt": record["prompt"],
        "ability": record["ability"],
        "reward_model": record["reward_model"],
        "extra_info": {
            "split": str(extra_info["split"]),
            "index": int(extra_info["index"]),
            "dataset": dataset,
            "domain": domain,
            "run_id": str(run_id),
            "pre_skill": None if pre_skill is None else str(pre_skill),
            "prompt_token_count": int(extra_info["prompt_token_count"]),
            "example_reward": (None if example_reward is None else float(example_reward)),
        },
    }


def get_record_run_id(record: dict) -> str | None:
    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict):
        return None

    run_id = extra_info.get("run_id")
    if run_id is None or run_id == "":
        return None

    return str(run_id)


def parse_run_id_index(run_id: str, id_prefix: str = RUN_ID_PREFIX) -> int | None:
    if not run_id.startswith(id_prefix):
        return None

    suffix = run_id[len(id_prefix) :]
    if not suffix.isdigit():
        return None

    return int(suffix)


def find_task_class(run_id_index: int) -> dict | None:
    for task_class in TASK_CLASSES:
        if task_class["start"] <= run_id_index <= task_class["end"]:
            return task_class
    return None


def select_uniform_records(items: list[tuple[int, int, dict]], keep_limit: int) -> list[dict]:
    if len(items) <= keep_limit:
        return [record for _run_id_index, _original_index, record in items]

    step = len(items) / keep_limit
    selected = []
    for i in range(keep_limit):
        item_index = int((i + 0.5) * step)
        selected.append(items[item_index][2])
    return selected


def apply_category_keep_limit(records: list[dict], keep_limit: int = CATEGORY_KEEP_LIMIT) -> list[dict]:
    grouped_records = {task_class["key"]: [] for task_class in TASK_CLASSES}
    uncategorized_run_ids = []

    for original_index, record in enumerate(records):
        run_id = get_record_run_id(record)
        run_id_index = parse_run_id_index(run_id) if run_id is not None else None
        task_class = find_task_class(run_id_index) if run_id_index is not None else None

        if task_class is None:
            uncategorized_run_ids.append(run_id)
            continue

        grouped_records[task_class["key"]].append((run_id_index, original_index, record))

    print(f"Records after reward/token filtering: {len(records)}")
    print("Before category cap:")
    for task_class in TASK_CLASSES:
        key = task_class["key"]
        print(f"{key}: {len(grouped_records[key])} ({task_class['range']}, {task_class['main_class']})")

    kept_records = []
    print("After category cap:")
    for task_class in TASK_CLASSES:
        key = task_class["key"]
        grouped_records[key].sort(key=lambda item: (item[0], item[1]))
        selected_records = select_uniform_records(grouped_records[key], keep_limit)
        kept_records.extend(selected_records)
        print(f"{key}: {len(selected_records)} ({task_class['range']}, {task_class['main_class']})")

    if uncategorized_run_ids:
        print(f"Dropped uncategorized records: {len(uncategorized_run_ids)}")
        print(f"Uncategorized run_ids: {uncategorized_run_ids}")

    return kept_records


def build_routed_records(
    routed_rollouts: list[RoutedRollout],
    *,
    split: str,
    bfcl_initial_contexts: dict[str, Any] | None,
    tau2_initial_contexts: dict[tuple[str, str], Any] | None,
) -> tuple[list[BuiltRecord], dict[str, int]]:
    built_records: list[BuiltRecord] = []
    skipped_by_dataset = {DATASET_BFCL: 0, DATASET_TAU2: 0}

    for routed in routed_rollouts:
        if routed.dataset == DATASET_BFCL:
            if bfcl_initial_contexts is None:
                raise ValueError("BFCL initial contexts were not loaded")
            record = phase.build_verl_record(
                routed.record,
                routed.source_index,
                split,
                bfcl_initial_contexts,
            )
            domain = None
        else:
            if tau2_initial_contexts is None:
                raise ValueError("Tau2 initial contexts were not loaded")
            record = build_tau2_verl_record(
                routed.record,
                routed.source_index,
                split,
                tau2_initial_contexts,
            )
            domain = str(routed.record["domain"])

        if record is None:
            skipped_by_dataset[routed.dataset] += 1
            continue

        normalized_record = normalize_verl_record_schema(
            record,
            dataset=routed.dataset,
            domain=domain,
        )
        built_records.append(BuiltRecord(routed.dataset, routed.source_index, normalized_record))

    return built_records, skipped_by_dataset


def apply_dataset_sampling(built_records: list[BuiltRecord]) -> list[dict[str, Any]]:
    bfcl_items = [item for item in built_records if item.dataset == DATASET_BFCL]
    tau2_items = [item for item in built_records if item.dataset == DATASET_TAU2]

    if not bfcl_items:
        return [item.record for item in tau2_items]

    selected_bfcl_records = apply_category_keep_limit([item.record for item in bfcl_items])
    item_by_record_id = {id(item.record): item for item in bfcl_items}
    selected_bfcl_items = [item_by_record_id[id(record)] for record in selected_bfcl_records]

    # Preserve the exact legacy BFCL category ordering for BFCL-only input.
    if not tau2_items:
        return [item.record for item in selected_bfcl_items]

    selected_bfcl_source_indexes = {item.source_index for item in selected_bfcl_items}
    return [
        item.record
        for item in built_records
        if item.dataset == DATASET_TAU2 or item.source_index in selected_bfcl_source_indexes
    ]


def build_train_features() -> datasets.Features:
    return datasets.Features(
        {
            "data_source": datasets.Value("string"),
            "prompt": [
                {
                    "role": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }
            ],
            "ability": datasets.Value("string"),
            "reward_model": {
                "style": datasets.Value("string"),
                "ground_truth": datasets.Value("string"),
            },
            "extra_info": {
                "split": datasets.Value("string"),
                "index": datasets.Value("int64"),
                "dataset": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "run_id": datasets.Value("string"),
                "pre_skill": datasets.Value("string"),
                "prompt_token_count": datasets.Value("int64"),
                "example_reward": datasets.Value("float64"),
            },
        }
    )


def count_by_dataset(routed_rollouts: list[RoutedRollout]) -> dict[str, int]:
    return {
        dataset: sum(item.dataset == dataset for item in routed_rollouts) for dataset in (DATASET_BFCL, DATASET_TAU2)
    }


def count_tau2_by_domain(routed_rollouts: list[RoutedRollout]) -> dict[str, int]:
    return {
        domain: sum(item.dataset == DATASET_TAU2 and item.record.get("domain") == domain for item in routed_rollouts)
        for domain in sorted(TAU2_DOMAINS)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed BFCL/Tau2 train.parquet from seed skill rollouts.")
    parser.add_argument(
        "--rollouts-path",
        default=str(DEFAULT_ORIGIN_DIR / "seed_skill_rollouts.jsonl"),
        help="BFCL, Tau2, or mixed seed rollout JSONL.",
    )
    parser.add_argument(
        "--initial-context-path",
        "--bfcl-initial-context-path",
        dest="bfcl_initial_context_path",
        default=os.getenv("BFCL_INITIAL_CONTEXT_QUERY_PATH", str(DEFAULT_INITIAL_CONTEXT_PATH)),
        help=(
            "BFCL initial_context/query JSONL used to recover system content by run_id. "
            "--initial-context-path is retained for backward compatibility."
        ),
    )
    parser.add_argument(
        "--tau2-initial-context-path",
        default=os.getenv(
            "TAU2_INITIAL_CONTEXT_QUERY_PATH",
            str(DEFAULT_TAU2_INITIAL_CONTEXT_PATH),
        ),
        help=("Tau2 initial_context/query JSONL used to recover system content by (domain, run_id)."),
    )
    parser.add_argument(
        "--output-train-file",
        default=str(DEFAULT_ORIGIN_DIR / "train.parquet"),
        help="Output parquet path for origin-stage skill optimization training.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--tokenizer-model-path",
        default=os.getenv("TOKENIZER_MODEL_PATH") or os.getenv("BASE_MODEL"),
        help="Optional tokenizer path override for prompt length filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path = Path(args.rollouts_path).expanduser()
    bfcl_initial_context_path = Path(args.bfcl_initial_context_path).expanduser()
    tau2_initial_context_path = Path(args.tau2_initial_context_path).expanduser()
    output_train_file = Path(args.output_train_file).expanduser()

    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts path does not exist: {rollouts_path}")

    maybe_patch_token_counter(phase, args.tokenizer_model_path)
    raw_records = list(phase.iter_jsonl_records(rollouts_path))
    if not raw_records:
        raise ValueError(f"No rollout records found in {rollouts_path}")
    routed_rollouts = route_rollouts(raw_records)
    input_counts = count_by_dataset(routed_rollouts)

    bfcl_initial_contexts = None
    if input_counts[DATASET_BFCL]:
        if not bfcl_initial_context_path.exists():
            raise FileNotFoundError(f"BFCL initial context path does not exist: {bfcl_initial_context_path}")
        bfcl_initial_contexts = phase.load_initial_context_by_run_id(bfcl_initial_context_path)

    tau2_initial_contexts = None
    if input_counts[DATASET_TAU2]:
        if not tau2_initial_context_path.exists():
            raise FileNotFoundError(f"Tau2 initial context path does not exist: {tau2_initial_context_path}")
        tau2_initial_contexts = load_tau2_initial_context_by_task(tau2_initial_context_path)

    built_records, skipped_by_dataset = build_routed_records(
        routed_rollouts,
        split=args.split,
        bfcl_initial_contexts=bfcl_initial_contexts,
        tau2_initial_contexts=tau2_initial_contexts,
    )
    if not built_records:
        raise ValueError(f"No train records were built from {rollouts_path}")

    records_after_build_filter = len(built_records)
    records = apply_dataset_sampling(built_records)
    if not records:
        raise ValueError(f"No train records remained after dataset sampling from {rollouts_path}")

    output_train_file.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = datasets.Dataset.from_list(
        records,
        features=build_train_features(),
    )
    train_dataset.to_parquet(str(output_train_file))

    final_counts = {
        dataset: sum(record["extra_info"]["dataset"] == dataset for record in records)
        for dataset in (DATASET_BFCL, DATASET_TAU2)
    }
    print(f"Train parquet saved to {output_train_file}")
    print(f"Total rollout records: {len(routed_rollouts)}")
    print(f"Input records by dataset: {input_counts}")
    if input_counts[DATASET_TAU2]:
        print(f"Tau2 input records by domain: {count_tau2_by_domain(routed_rollouts)}")
    print(f"Kept train records before category cap: {records_after_build_filter}")
    print(f"Kept train records after category cap: {len(records)}")
    print(f"Final records by dataset: {final_counts}")
    print(f"Skipped records during build_verl_record: {sum(skipped_by_dataset.values())} ({skipped_by_dataset})")


if __name__ == "__main__":
    main()
