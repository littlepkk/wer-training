from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner_helper import *
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.utils import *


def get_handler(model_name: str) -> BaseHandler:
    config = MODEL_CONFIG_MAPPING[model_name]
    handler: BaseHandler = config.model_handler(
        model_name=config.model_name,
        temperature=0,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )
    return handler


def _subset_entries_by_model_ids(
    model_result_entries: list[dict],
    prompt_entries: list[dict],
    ground_truth_entries: list[dict] = None,  # Irrelevance entries don't have ground truth
    allow_missing: bool = False,
):
    """
    Filter the prompt and ground truth entries so that its order/length matches the IDs present in `model_result`. When `allow_missing` is False, all IDs must be present; otherwise, any missing IDs are silently ignored.
    """
    if not model_result_entries:
        return [], []

    if not allow_missing and (len(model_result_entries) != len(prompt_entries)):
        raise ValueError(
            f"Length of model result ({len(model_result_entries)}) does not match length of test entries ({len(prompt_entries)}). If you intended to run only on a subset (eg. entries present in the model result), please pass the `--partial-eval` flag."
        )

    all_present_ids = {entry["id"]: entry for entry in model_result_entries}

    # Align prompt and ground-truth using the *index* of the prompt entry. Some
    # ground-truth items use a different ID format, but the order between the
    # prompt list and the ground-truth list is guaranteed to be identical. We
    # therefore keep the element at index *i* in both lists whenever the
    # prompt entry at that index has an ID present in the model results.
    filtered_prompt_entries: list[dict] = []
    filtered_ground_truth_entries: list[dict] = []
    for idx, prompt_entry in enumerate(prompt_entries):
        if prompt_entry["id"] in all_present_ids:
            filtered_prompt_entries.append(prompt_entry)
            # ground_truth_entries and prompt_entries are aligned by index.
            if ground_truth_entries is not None:
                filtered_ground_truth_entries.append(ground_truth_entries[idx])

    return filtered_prompt_entries, filtered_ground_truth_entries


def _evaluate_single_multi_turn_entry(
    handler: BaseHandler,
    test_entry_id,
    model_result_list,
    ground_truth_list,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single multi-turn entry."""
    # Remove the function doc from the score file for better readability
    if "function" in prompt_entry:
        del prompt_entry["function"]

    if type(model_result_list) != list:
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    "Error during inference phase. Model did not output a list of model responses."
                ],
                "error_type": "multi_turn:inference_error",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    # Check if force-terminated during inference phase.
    # This happens when the model has retried too many times and still haven't figured out the answer.
    # When force-terminated, no further evaluation is needed. This whole entry will be failed.
    if len(model_result_list) != len(ground_truth_list):
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    f"Model was force-terminated during inference phase. The length of the model result turns ({len(model_result_list)}) does not match the length of the ground truth turns ({len(ground_truth_list)})."
                ],
                "error_type": "multi_turn:force_terminated",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    # decode_execute returns a list of strings
    multi_turn_model_result_list_decoded: list[list[list[str]]] = []
    # Try decoding the model results into executable function calls
    for single_turn_model_result_list in model_result_list:
        single_turn_model_result_list_decoded = []
        for model_result_item in single_turn_model_result_list:
            # model_result_item is per step
            try:
                decoded_result: list[str] = handler.decode_execute(
                    model_result_item, has_tool_call_tag=False
                )
                if is_empty_execute_response(decoded_result):
                    # Empty output is not considered as a valid function call
                    continue
                single_turn_model_result_list_decoded.append(decoded_result)
            except Exception as e:
                # Ignore any failed decoding and continue to the next message
                # We only care about the decoded function call, not the error message or if the model is chatting
                continue
        # pdb.set_trace()
        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)
    # pdb.set_trace()
    # Check if the model output the correct function calls
    accuracy_checker_result = multi_turn_checker(
        multi_turn_model_result_list_decoded,
        ground_truth_list,
        prompt_entry,
        test_category,
        model_name,
    )

    if not accuracy_checker_result["valid"]:
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": accuracy_checker_result.pop("valid"),
            "error": accuracy_checker_result,
            "prompt": prompt_entry,
            "model_result_raw": model_result_list,
            "model_result_decoded": multi_turn_model_result_list_decoded,
            "possible_answer": ground_truth_list,
        }

    return {"valid": True}


def multi_turn_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
    score_run_id=None,
    score_sequence_1=None,
    score_sequence_2=None,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        multi_turn_model_result_list = model_result[i]["result"]
        multi_turn_ground_truth_list = possible_answer[i]["ground_truth"]
        test_entry = prompt[i]

        entry_result = _evaluate_single_multi_turn_entry(
            handler,
            index,
            multi_turn_model_result_list,
            multi_turn_ground_truth_list,
            test_entry,
            model_name,
            test_category,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)

    return save_eval_results(
        result,
        correct_count,
        model_result,
        test_category,
        model_name,
        score_dir,
        score_run_id=score_run_id,
        score_sequence_1=score_sequence_1,
        score_sequence_2=score_sequence_2,
    )


def evaluate_task(test_category, result_dir, score_dir, model_result, model_name,
                  handler, leaderboard_table, allow_missing=False,
                  score_run_id=None, score_sequence_1=None, score_sequence_2=None):
    if not is_multi_turn(test_category):
        raise ValueError("WER training only supports BFCL multi-turn reward tasks")
    record_cost_latency(leaderboard_table, model_name, model_result)
    prompt = load_dataset_entry(test_category, include_prereq=False,
                                include_language_specific_hint=False)
    possible_answer = load_ground_truth_entry(test_category)
    assert len(prompt) == len(possible_answer)
    prompt, possible_answer = _subset_entries_by_model_ids(
        model_result, prompt, possible_answer, allow_missing=allow_missing)
    accuracy, total_count = multi_turn_runner(
        handler, model_result, prompt, possible_answer, model_name, test_category,
        score_dir, score_run_id=score_run_id, score_sequence_1=score_sequence_1,
        score_sequence_2=score_sequence_2)
    record_result(leaderboard_table, model_name, test_category, accuracy, total_count)
    return leaderboard_table
