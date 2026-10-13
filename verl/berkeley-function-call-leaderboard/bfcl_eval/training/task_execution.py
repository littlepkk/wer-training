import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
# import pdb
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from bfcl_eval.training.generation import generate_results
from bfcl_eval.constants.eval_config import DOTENV_PATH
from bfcl_eval.training.task_runtime import (
    build_generation_args,
    configure_runtime,
    load_single_result,
    prepare_test_cases,
    resolve_result_dir,
    validate_model,
)


FIXED_MODEL = "local_fc"
FIXED_ALLOW_OVERWRITE = True




def inject_skill_system_prompt(test_cases: list[dict], skill: str | None) -> None:
    if not skill:
        return

    skill = skill.strip()
    if not skill:
        return

    for test_case in test_cases:
        first_turn = test_case["question"][0]
        if first_turn and first_turn[0]["role"] == "system":
            first_turn[0]["content"] += "\n\n" + "This is a skill; you can refer to its procedures or rules to help you complete the task." + skill
        else:
            first_turn.insert(0, {"role": "system", "content": "This is a skill; you can refer to its procedures or rules to help you complete the task: "+ skill})


def run_single_task_inference(
    run_id: str,
    result_dir: str | None,
    sequence_1: str | int,
    sequence_2: str | int,
    skill: str | None = None,
) -> dict:
    args = SimpleNamespace(
        model=FIXED_MODEL,
        run_id=run_id,
        result_run_id=run_id,
        result_sequence_1=sequence_1,
        result_sequence_2=sequence_2,
        temperature=0.001,
        result_dir=result_dir,
        num_threads=None,
        include_input_log=False,
        exclude_state_log=False,
        allow_overwrite=FIXED_ALLOW_OVERWRITE,
        skill=skill,
    )

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    configure_runtime()
    validate_model(args.model)

    result_dir = resolve_result_dir(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    test_cases = prepare_test_cases(
        run_id=args.run_id,
        model_name=args.model,
        result_dir=result_dir,
        allow_overwrite=args.allow_overwrite,
        result_run_id=args.result_run_id,
        result_sequence_1=args.result_sequence_1,
        result_sequence_2=args.result_sequence_2,
    )

    if test_cases:
        inject_skill_system_prompt(test_cases, args.skill)
        generation_args = build_generation_args(args, result_dir)
        generate_results(generation_args, args.model, test_cases)

    return load_single_result(
        result_dir,
        args.model,
        args.run_id,
        result_run_id=args.result_run_id,
        result_sequence_1=args.result_sequence_1,
        result_sequence_2=args.result_sequence_2,
    )


