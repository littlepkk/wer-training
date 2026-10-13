import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    PROJECT_ROOT,
    RESULT_PATH,
    SCORE_PATH,
)
from bfcl_eval.training.checker import evaluate_task, get_handler
from bfcl_eval.utils import (
    extract_test_category_from_id,
    get_directory_structure_by_category,
    get_directory_structure_by_id,
    load_file,
)


FIXED_MODEL = "local_fc"


def _resolve_result_dir(result_dir: str | None) -> Path:
    if result_dir is None:
        return RESULT_PATH
    return (PROJECT_ROOT / result_dir).resolve()


def _resolve_score_dir(score_dir: str | None) -> Path:
    if score_dir is None:
        return SCORE_PATH
    return (PROJECT_ROOT / score_dir).resolve()


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def get_single_task_result_file_path(
    result_dir: Path,
    model_name: str,
    run_id: str,
    sequence_1: str | int,
    sequence_2: str | int,
) -> Path:
    test_category = extract_test_category_from_id(run_id)
    return (
        result_dir
        / _model_dir_name(model_name)
        / get_directory_structure_by_id(run_id)
        / str(run_id)
        / f"{VERSION_PREFIX}_{test_category}_{sequence_1}_{sequence_2}_result.json"
    )


def get_single_task_score_file_path(
    score_dir: Path,
    model_name: str,
    run_id: str,
    test_category: str,
    sequence_1: str | int,
    sequence_2: str | int,
) -> Path:
    return (
        score_dir
        / _model_dir_name(model_name)
        / get_directory_structure_by_category(test_category)
        / str(run_id)
        / f"{VERSION_PREFIX}_{test_category}_{sequence_1}_{sequence_2}_score.json"
    )


def evaluate_single_task_result(
    run_id: str,
    sequence_1: str | int,
    sequence_2: str | int,
    result_dir: str | None = None,
    score_dir: str | None = None,
) -> float:
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)

    resolved_result_dir = _resolve_result_dir(result_dir)
    resolved_score_dir = _resolve_score_dir(score_dir)
    result_file_path = get_single_task_result_file_path(
        resolved_result_dir,
        FIXED_MODEL,
        run_id,
        sequence_1,
        sequence_2,
    )
    if not result_file_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_file_path}")

    model_result = load_file(result_file_path, sort_by_id=True)
    if not model_result:
        raise ValueError(f"Result file is empty: {result_file_path}")

    target_entry = next((entry for entry in model_result if entry["id"] == run_id), None)
    if target_entry is None:
        raise ValueError(f"Run id '{run_id}' was not found in {result_file_path}")

    test_category = extract_test_category_from_id(target_entry["id"], remove_prereq=True)
    model_dir_name = _model_dir_name(FIXED_MODEL)
    handler = get_handler(FIXED_MODEL)
    leaderboard_table = {}

    evaluate_task(
        test_category=test_category,
        result_dir=resolved_result_dir,
        score_dir=resolved_score_dir,
        model_result=model_result,
        model_name=model_dir_name,
        handler=handler,
        leaderboard_table=leaderboard_table,
        allow_missing=True,
        score_run_id=run_id,
        score_sequence_1=sequence_1,
        score_sequence_2=sequence_2,
    )

    score_file_path = get_single_task_score_file_path(
        resolved_score_dir,
        FIXED_MODEL,
        run_id,
        test_category,
        sequence_1,
        sequence_2,
    )
    score_summary = leaderboard_table[model_dir_name][test_category]

    # return {
    #     "run_id": run_id,
    #     "sequence_1": sequence_1,
    #     "sequence_2": sequence_2,
    #     "test_category": test_category,
    #     "accuracy": score_summary["accuracy"],
    #     "total_count": score_summary["total_count"],
    #     "result_file": str(result_file_path),
    #     "score_file": str(score_file_path),
    # }
    return score_summary["accuracy"]
    
