import argparse
import heapq
import multiprocessing as mp
import os
import queue
import shutil
import threading
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Optional

from bfcl_eval.constants.eval_config import (
    PROJECT_ROOT,
    RESULT_FILE_PATTERN,
    RESULT_PATH,
    TEST_IDS_TO_GENERATE_PATH,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.utils import *
from tqdm import tqdm




def build_handler(model_name, temperature):
    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(
        model_name=config.model_name,
        temperature=temperature,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )
    return handler






def multi_threaded_inference(handler, test_case, include_input_log, exclude_state_log):

    assert type(test_case["function"]) is list

    try:
        result, metadata = handler.inference(
            test_case, include_input_log, exclude_state_log
        )
    except Exception as e:
        # This is usually the case when the model getting stuck on one particular test case.
        # For example, timeout error or FC model returning invalid JSON response.
        # Since temperature is already set to 0.001, retrying the same test case will not help.
        # So we continue the generation process and record the error message as the model response
        error_block = (
            "-" * 100
            + "\n❗️❗️ Error occurred during inference. Continuing to next test case.\n"
            + f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}\n"
            + traceback.format_exc(limit=10)
            + "-" * 100
        )
        tqdm.write(error_block)

        result = f"Error during inference: {str(e)}"
        metadata = {"traceback": traceback.format_exc()}

    result_to_write = {
        "id": test_case["id"],
        "result": result,
        **metadata,
    }
    
    return result_to_write


def generate_results(args, model_name, test_cases_total):
    handler = build_handler(model_name, args.temperature)

    if isinstance(handler, OSSHandler):
        handler: OSSHandler
        is_oss_model = True
        # For OSS models, if the user didn't explicitly set the number of threads,
        # we default to 100 threads to speed up the inference.
        num_threads = (
            args.num_threads
            if args.num_threads is not None
            else LOCAL_SERVER_MAX_CONCURRENT_REQUEST
        )
    else:
        handler: BaseHandler
        is_oss_model = False
        num_threads = args.num_threads if args.num_threads is not None else 1

    # Use a separate thread to write the results to the file to avoid concurrent IO issues
    def _writer():
        """Consume result dicts from the queue and write them with exclusive access."""
        while True:
            item = write_queue.get()
            if item is None:
                break
            handler.write(
                item,
                result_dir=args.result_dir,
                update_mode=args.run_ids,
                result_run_id=getattr(args, "result_run_id", None),
                result_sequence_1=getattr(args, "result_sequence_1", None),
                result_sequence_2=getattr(args, "result_sequence_2", None),
            )
            write_queue.task_done()

    write_queue: queue.Queue = queue.Queue()

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    try:
        if is_oss_model:
            handler.spin_up_local_server(
                num_gpus=args.num_gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
                backend=args.backend,
                skip_server_setup=args.skip_server_setup,
                local_model_path=args.local_model_path,
                lora_modules=args.lora_modules,
                enable_lora=args.enable_lora,
                max_lora_rank=args.max_lora_rank,
            )

        # ───── dependency bookkeeping ──────────────────────────────
        dependencies = {
            test_case["id"]: set(test_case.get("depends_on", []))
            for test_case in test_cases_total
        }
        children_of = defaultdict(list)
        for test_case in test_cases_total:
            for dependency_id in test_case.get("depends_on", []):
                children_of[dependency_id].append(test_case["id"])

        id_to_test_case = {test_case["id"]: test_case for test_case in test_cases_total}

        ready_queue = [
            (sort_key(id_to_test_case[test_case_id]), test_case_id)
            for test_case_id, dependency_ids in dependencies.items()
            if not dependency_ids
        ]
        heapq.heapify(ready_queue)
        in_flight: dict[Future, str] = {}  # future -> test_case_id
        completed = set()

        with ThreadPoolExecutor(max_workers=num_threads) as pool, tqdm(
            total=len(test_cases_total),
            desc=f"Generating results for {model_name}",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:

            # seed initial ready tasks
            while ready_queue and len(in_flight) < num_threads:
                _, test_case_id = heapq.heappop(ready_queue)
                test_case = id_to_test_case[test_case_id]
                future = pool.submit(
                    multi_threaded_inference,
                    handler,
                    test_case,
                    args.include_input_log,
                    args.exclude_state_log,
                )
                in_flight[future] = test_case_id

            # main scheduler loop
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    test_case_id = in_flight.pop(future)
                    result_dict = future.result()

                    # Enqueue the result for the writer thread to handle file IO
                    write_queue.put(result_dict)

                    # Update progress bar right after inference completes
                    pbar.update()
                    completed.add(test_case_id)

                    # unlock children
                    for child_id in children_of[test_case_id]:
                        dependencies[child_id].discard(test_case_id)
                        if not dependencies[child_id]:
                            heapq.heappush(
                                ready_queue,
                                (sort_key(id_to_test_case[child_id]), child_id),
                            )

                # refill the pool up to max_workers
                while ready_queue and len(in_flight) < num_threads:
                    _, test_case_id = heapq.heappop(ready_queue)
                    test_case = id_to_test_case[test_case_id]
                    future = pool.submit(
                        multi_threaded_inference,
                        handler,
                        test_case,
                        args.include_input_log,
                        args.exclude_state_log,
                    )
                    in_flight[future] = test_case_id

    finally:
        # Signal writer thread to finish and wait for it
        write_queue.put(None)
        writer_thread.join()

        if is_oss_model:
            handler.shutdown_local_server()


