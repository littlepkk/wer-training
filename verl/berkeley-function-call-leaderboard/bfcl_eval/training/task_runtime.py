import argparse
import json
import multiprocessing as mp
import os
import sys
import pdb
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from bfcl_eval.training.generation import generate_results
from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.eval_config import DOTENV_PATH, PROJECT_ROOT, RESULT_PATH
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.utils import (
    extract_test_category_from_id,
    get_directory_structure_by_id,
    is_format_sensitivity,
    load_dataset_entry,
    load_file,
    populate_initial_settings_for_memory_test_cases,
    populate_initial_settings_for_web_search_test_cases,
    sort_key,
)


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------



def configure_runtime() -> None:
    
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    mp.set_start_method("spawn", force=True)


def resolve_result_dir(result_dir: str | None) -> Path:
    
    if result_dir is None:
        return RESULT_PATH
    return (PROJECT_ROOT / result_dir).resolve()


def validate_model(model_name: str) -> None:
    
    
    if model_name not in MODEL_CONFIG_MAPPING:
        raise ValueError(f"Unknown model_name '{model_name}'.")

    handler_cls = MODEL_CONFIG_MAPPING[model_name].model_handler
    if ".api_inference." not in handler_cls.__module__:
        raise ValueError(
            f"Model '{model_name}' is not an api_inference model. "
            "This script only supports api_inference handlers."
        )


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_result_file_path(
    result_dir: Path,
    model_name: str,
    entry_id: str,
    result_run_id: str | None = None,
    result_sequence_1: str | int | None = None,
    result_sequence_2: str | int | None = None,
) -> Path:
    
    
    model_dir = result_dir / model_name.replace("/", "_")
    group_dir = model_dir / get_directory_structure_by_id(entry_id)
    test_category = extract_test_category_from_id(entry_id)
    if (
        result_run_id is None
        and result_sequence_1 is None
        and result_sequence_2 is None
    ):
        return group_dir / f"{VERSION_PREFIX}_{test_category}_result.json"

    if (
        result_run_id is None
        or result_sequence_1 is None
        or result_sequence_2 is None
    ):
        raise ValueError(
            "result_run_id, result_sequence_1, and result_sequence_2 must be "
            "provided together."
        )

    run_dir = group_dir / str(result_run_id)
    return (
        run_dir
        / f"{VERSION_PREFIX}_{test_category}_{result_sequence_1}_{result_sequence_2}_result.json"
    )


def load_dataset_entries_by_id(run_id: str) -> tuple[str, dict[str, dict]]:
    
    
    dataset_category = extract_test_category_from_id(run_id, remove_prereq=True)
    all_entries = load_dataset_entry(dataset_category)
    entries_by_id = {entry["id"]: deepcopy(entry) for entry in all_entries}
    if run_id not in entries_by_id:
        raise ValueError(
            f"Run id '{run_id}' was not found in dataset category '{dataset_category}'."
        )
    return dataset_category, entries_by_id


def collect_dependency_closure(
    target_run_id: str, entries_by_id: dict[str, dict]
) -> list[dict]:
    
    
    selected_ids: set[str] = set()
    pending = [target_run_id]

    while pending:
        current_id = pending.pop()
        if current_id in selected_ids:
            continue
        if current_id not in entries_by_id:
            raise ValueError(
                f"Dependency run id '{current_id}' was not found while resolving "
                f"'{target_run_id}'."
            )
        selected_ids.add(current_id)
        pending.extend(entries_by_id[current_id].get("depends_on", []))
    return [deepcopy(entries_by_id[entry_id]) for entry_id in selected_ids]


def load_existing_ids_for_entries(
    result_dir: Path,
    model_name: str,
    entries: list[dict],
    result_run_id: str | None = None,
    result_sequence_1: str | int | None = None,
    result_sequence_2: str | int | None = None,
) -> set[str]:
    
    
    existing_ids: set[str] = set()
    file_paths = {
        get_result_file_path(
            result_dir,
            model_name,
            entry["id"],
            result_run_id=result_run_id,
            result_sequence_1=result_sequence_1,
            result_sequence_2=result_sequence_2,
        )
        for entry in entries
    }
    for file_path in file_paths:
        if not file_path.exists():
            continue
        existing_ids.update(entry["id"] for entry in load_file(file_path))
    return existing_ids


def filter_generation_entries(
    run_id: str,
    entries: list[dict],
    existing_ids: set[str],
    allow_overwrite: bool,
) -> list[dict]:
    
    
    if not allow_overwrite and run_id in existing_ids:
        return []

    if allow_overwrite:
        entries_to_generate = entries
    else:
        entries_to_generate = [entry for entry in entries if entry["id"] not in existing_ids]

    entries_to_generate_ids = {entry["id"] for entry in entries_to_generate}
    for entry in entries_to_generate:
        if "depends_on" in entry:
            entry["depends_on"] = [
                dep_id
                for dep_id in entry["depends_on"]
                if dep_id in entries_to_generate_ids
            ]

    return entries_to_generate


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def prepare_test_cases(
    run_id: str,
    model_name: str,
    result_dir: Path,
    allow_overwrite: bool,
    result_run_id: str | None = None,
    result_sequence_1: str | int | None = None,
    result_sequence_2: str | int | None = None,
) -> list[dict]:
    
    
    dataset_category, entries_by_id = load_dataset_entries_by_id(run_id)

    if (
        is_format_sensitivity(dataset_category)
        and MODEL_CONFIG_MAPPING[model_name].is_fc_model
    ):
        raise ValueError(
            "Format sensitivity test cases are only supported for prompting (non-FC) models."
        )
    selected_entries = collect_dependency_closure(run_id, entries_by_id)
    existing_ids = load_existing_ids_for_entries(
        result_dir,
        model_name,
        selected_entries,
        result_run_id=result_run_id,
        result_sequence_1=result_sequence_1,
        result_sequence_2=result_sequence_2,
    )
    entries_to_generate = filter_generation_entries(
        run_id,
        selected_entries,
        existing_ids,
        allow_overwrite,
    )

    model_result_dir = result_dir / model_name.replace("/", "_")
    entries_to_generate = populate_initial_settings_for_memory_test_cases(
        entries_to_generate, model_result_dir
    )
    entries_to_generate = populate_initial_settings_for_web_search_test_cases(
        entries_to_generate
    )

    return sorted(entries_to_generate, key=sort_key)


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_generation_args(args: argparse.Namespace, result_dir: Path) -> SimpleNamespace:
    
    
    return SimpleNamespace(
        temperature=args.temperature,
        include_input_log=args.include_input_log,
        exclude_state_log=args.exclude_state_log,
        num_threads=args.num_threads,
        num_gpus=1,
        gpu_memory_utilization=0.9,
        backend="sglang",
        skip_server_setup=True,
        local_model_path=None,
        lora_modules=None,
        enable_lora=False,
        max_lora_rank=None,
        result_dir=result_dir,
        result_run_id=getattr(args, "result_run_id", None),
        result_sequence_1=getattr(args, "result_sequence_1", None),
        result_sequence_2=getattr(args, "result_sequence_2", None),
        run_ids=True,
    )


def load_single_result(
    result_dir: Path,
    model_name: str,
    run_id: str,
    result_run_id: str | None = None,
    result_sequence_1: str | int | None = None,
    result_sequence_2: str | int | None = None,
) -> dict:
    
    
    file_path = get_result_file_path(
        result_dir,
        model_name,
        run_id,
        result_run_id=result_run_id,
        result_sequence_1=result_sequence_1,
        result_sequence_2=result_sequence_2,
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Result file not found for run id '{run_id}': {file_path}")

    for entry in load_file(file_path, sort_by_id=True):
        if entry["id"] == run_id:
            return entry

    raise ValueError(f"Run id '{run_id}' was not found in result file '{file_path}'.")




