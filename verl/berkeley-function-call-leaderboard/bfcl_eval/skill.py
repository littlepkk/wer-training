import math
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from transformers import AutoTokenizer

from bfcl_eval.training.task_execution import run_single_task_inference
from bfcl_eval.training.task_reward import evaluate_single_task_result


NUM_PARALLEL_TESTS = 2
MAX_DYNAMIC_STEP = 125
FORMAT_REWARD_START = 1.5
FORMAT_REWARD_END = 1.0
TARGET_FAILED_SKILL_REWARD = 1.0
REWARD_TOLERANCE = 0.5
FAILED_SKILL_LOG_PATH = (
    Path(os.getenv("WER_DATA_DIR", str(Path(__file__).resolve().parents[3] / "extra_datasets")))
    / "{phase_id}"
    / "failed_skill_records.jsonl"
)
_FAILED_SKILL_LOG_LOCK = threading.Lock()

TAU2_ROOT = Path(__file__).resolve().parents[2] / "tau2-bench-verified"
TAU2_SRC_ROOT = TAU2_ROOT / "src"

TAU2_TRIAL_SEEDS = (300, 301)
TAU2_TRIAL_MAX_CONCURRENCY = 2
TAU2_MAX_STEPS = 23
TAU2_MAX_ERRORS = 10
TAU2_AGENT_MODEL = os.getenv("TAU2_AGENT_MODEL", f"openai/{os.getenv('WER_EXECUTOR_MODEL', 'executor-model')}")
TAU2_AGENT_API_KEY = os.getenv("TAU2_AGENT_API_KEY", os.getenv("WER_EXECUTOR_API_KEY", "EMPTY"))
TAU2_AGENT_API_BASE = os.getenv("TAU2_AGENT_API_BASE", os.getenv("WER_EXECUTOR_BASE_URL", "http://localhost:8000/v1"))
TAU2_AGENT_REQUEST_TIMEOUT = 300.0
TAU2_AGENT_REQUEST_RETRIES = 3
TAU2_AGENT_REQUEST_RETRY_SLEEP = 7.0
TAU2_USER_MODEL = os.getenv("TAU2_USER_MODEL", "openai/user-simulator-model")
TAU2_USER_API_KEY = os.getenv("TAU2_USER_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))

local_model_path = os.getenv("LOCAL_MODEL_PATH") or os.getenv("BASE_MODEL", "/path/to/models/skill-optimizer")


_tokenizer = None


def get_tokenizer(model_path=None):
    
    global _tokenizer
    tokenizer_path = model_path or local_model_path
    if _tokenizer is None and tokenizer_path:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return _tokenizer


def _get_dynamic_step(extra_info, name):
    
    extra_info = extra_info or {}
    trajectory_info = extra_info.get("__trajectory_info__", {})
    if not isinstance(trajectory_info, dict):
        trajectory_info = {}
    step = trajectory_info.get("global_step", extra_info.get("step", 10000))
    if step is None:
        print("step检查为none，后续需要进一步排查")
        step = 0
    
    step = max(0, step - 1)
    step = max(0, min(step, MAX_DYNAMIC_STEP))
    print(f"当前{name}_step:", step)
    return step



def format_reward(data_source, solution_str, ground_truth, extra_info):
    step = _get_dynamic_step(extra_info, "format")
    ratio = step / MAX_DYNAMIC_STEP if MAX_DYNAMIC_STEP > 0 else 1.0

    
    base_max = FORMAT_REWARD_START + (FORMAT_REWARD_END - FORMAT_REWARD_START) * ratio
    base_min = 0.0
    max_possible_reward = base_max
    min_possible_reward = base_min

    pattern = r"^<think>\n.*?\n</think>\n\n<skill>\n.*?\n</skill>$"

    completions = [solution_str] if isinstance(solution_str, str) else solution_str

    rewards = []
    for content in completions:
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        rewards.append(max_possible_reward if match else min_possible_reward)

    print("Format rewards:", rewards)
    return rewards


def extract_skill(text):
    
    if not isinstance(text, str):
        return ""
    m = re.search(r"<skill>\n(.*?)\n</skill>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""



def _score_single_parallel_test(skill, run_id, result_dir, score_dir, sequence_1, sequence_2):
    result_entry = None
    try:
        result_entry = run_single_task_inference(
            run_id=run_id,
            result_dir=result_dir,
            sequence_1=sequence_1,
            sequence_2=sequence_2,
            skill=skill,
        )
        reward = float(
            evaluate_single_task_result(
                run_id=run_id,
                sequence_1=sequence_1,
                sequence_2=sequence_2,
                result_dir=result_dir,
                score_dir=score_dir,
            )
        )
        return {
            "sequence_2": sequence_2,
            "reward": reward,
            "trajectory": result_entry.get("inference_log"),
        }
    except Exception as e:
        print(
            "skill acc_reward failed: "
            f"run_id={run_id}, sequence_1={sequence_1}, sequence_2={sequence_2}, error={e}"
        )
        return {
            "sequence_2": sequence_2,
            "reward": 0.0,
            "trajectory": result_entry.get("inference_log") if isinstance(result_entry, dict) else None,
            "error": str(e),
        }


def _get_failed_skill_log_path(extra_info):
    extra_info = extra_info or {}
    phase_id = extra_info.get("train_phase", 0)
    return Path(str(FAILED_SKILL_LOG_PATH).format(phase_id=phase_id))


def _get_pre_skill_index(extra_info):
    extra_info = extra_info or {}
    return str(extra_info.get("pre_skill", "origin"))


def _append_failed_skill_record(record, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _FAILED_SKILL_LOG_LOCK:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(line)


def _print_saved_skill_record(dataset, run_id, reward):
    print(
        f"[skill-record saved] dataset={dataset} id={run_id} reward={reward:g}",
        flush=True,
    )


def _acc_reward_bfcl(skill, extra_info):
    run_id = extra_info["run_id"]
    traj = extra_info["__trajectory_info__"]
    rollout_n = traj["rollout_n"]
    epoch_id = traj.get("epoch", 0)
    phase_id = extra_info.get("train_phase", 0)
    pre_skill_index = _get_pre_skill_index(extra_info)
    sequence_1 = f"phase{phase_id}_epoch{epoch_id}_{pre_skill_index}_skill{rollout_n}"
    sequence_2_values = [f"traj{i}" for i in range(NUM_PARALLEL_TESTS)]
    failed_skill_log_path = _get_failed_skill_log_path(extra_info)
    # result_dir = extra_info.get("result_dir", "result/skill_reward")
    # score_dir = extra_info.get("score_dir")

    with ThreadPoolExecutor(max_workers=NUM_PARALLEL_TESTS) as executor:
        test_results = list(
            executor.map(
                lambda sequence_2: _score_single_parallel_test(
                    skill,
                    run_id,
                    None,
                    None,
                    sequence_1,
                    sequence_2,
                ),
                sequence_2_values,
            )
        )

    reward = sum(item["reward"] for item in test_results)
    if abs(reward - TARGET_FAILED_SKILL_REWARD) < REWARD_TOLERANCE:
        _append_failed_skill_record(
            {
                "run_id": run_id,
                "skill": skill,
                "sequence_1": sequence_1,
                "reward": reward,
                "trajectories": test_results,
            },
            failed_skill_log_path,
        )
        _print_saved_skill_record("bfcl", run_id, reward)
    else:
        print("skip it!")
    return reward


def _load_tau2_evaluator():
    
    for project_path in (str(TAU2_ROOT), str(TAU2_SRC_ROOT)):
        if project_path not in sys.path:
            sys.path.insert(0, project_path)

    from tau2.training.task_reward import evaluate_tau2_task

    return evaluate_tau2_task


def _tau2_llm_args(
    *,
    api_key,
    api_base,
    temperature,
    max_tokens,
    timeout,
    num_retries,
    retry_sleep,
):
    
    args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "num_retries": num_retries,
        "tau2_retry_sleep": retry_sleep,
    }
    if api_key:
        args["api_key"] = api_key
    if api_base:
        args["api_base"] = api_base
    return args


def _should_keep_tau2_record(reward):
    return 0.5 <= reward <= 1.5


def _acc_reward_tau2(skill, extra_info):
    domain = extra_info.get("domain")
    run_id = extra_info.get("run_id")
    if not isinstance(domain, str) or not domain:
        print(f"tau2 acc_reward failed: invalid domain={domain!r}")
        return 0.0
    if run_id is None or run_id == "":
        print(f"tau2 acc_reward failed: invalid run_id={run_id!r}")
        return 0.0
    run_id = str(run_id)

    try:
        evaluate_tau2_task = _load_tau2_evaluator()
        result = evaluate_tau2_task(
            domain=domain,
            task_id=run_id,
            task_split_name="base",
            agent="llm_agent",
            agent_llm=TAU2_AGENT_MODEL,
            agent_llm_args=_tau2_llm_args(
                api_key=TAU2_AGENT_API_KEY,
                api_base=TAU2_AGENT_API_BASE,
                temperature=0.3,
                max_tokens=4096,
                timeout=TAU2_AGENT_REQUEST_TIMEOUT,
                num_retries=TAU2_AGENT_REQUEST_RETRIES,
                retry_sleep=TAU2_AGENT_REQUEST_RETRY_SLEEP,
            ),
            user="user_simulator",
            user_llm=TAU2_USER_MODEL,
            user_llm_args=_tau2_llm_args(
                api_key=TAU2_USER_API_KEY,
                api_base=os.getenv("TAU2_USER_API_BASE") or os.getenv("DEEPSEEK_API_BASE"),
                temperature=0.0,
                max_tokens=4096,
                timeout=float(os.getenv("TAU2_USER_REQUEST_TIMEOUT", "300")),
                num_retries=int(os.getenv("TAU2_USER_REQUEST_RETRIES", "2")),
                retry_sleep=float(os.getenv("TAU2_USER_REQUEST_RETRY_SLEEP", "2")),
            ),
            trial_seeds=TAU2_TRIAL_SEEDS,
            max_concurrency=TAU2_TRIAL_MAX_CONCURRENCY,
            max_steps=TAU2_MAX_STEPS,
            max_errors=TAU2_MAX_ERRORS,
            skill=skill,
        )
    except Exception as e:
        
        print(f"tau2 acc_reward failed: domain={domain}, run_id={run_id}, error={e}")
        return 0.0

    reward = float(result.acc_reward)
    if _should_keep_tau2_record(reward):
        _append_failed_skill_record(
            {
                "domain": domain,
                "run_id": run_id,
                "skill": skill,
                "reward": reward,
                "trajectories": [
                    {
                        "reward": float(trial.reward),
                        "trajectory": trial.trajectory,
                    }
                    for trial in result.trials
                ],
            },
            _get_failed_skill_log_path(extra_info),
        )
        _print_saved_skill_record(f"tau2bench/{domain}", run_id, reward)
    else:
        print("skip tau2 record!")
    return reward


def acc_reward(data_source, solution_str, ground_truth, extra_info):
    skill = extract_skill(solution_str)
    if not skill:
        print("skill acc_reward failed: missing or empty <skill> block")
        return 0.0

    extra_info = extra_info or {}
    dataset = extra_info.get("dataset", "bfcl")
    if dataset == "bfcl":
        return _acc_reward_bfcl(skill, extra_info)
    if dataset == "tau2":
        return _acc_reward_tau2(skill, extra_info)
    raise ValueError(f"Unsupported dataset for skill acc_reward: {dataset!r}")



def extract_think(text):
    if not isinstance(text, str):
        return ""
    m = re.search(r"<think>\n(.*?)\n</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    return ""


def _effective_len_zero(think: str) -> bool:
    
    s = re.sub(r"\s+", "", think)
    s = s.replace("<think>", "").replace("</think>", "")
    return len(s) == 0


def len_reward(data_source, solution_str, ground_truth, extra_info):
    tok = get_tokenizer()
    if tok is None:
        raise ValueError("Tokenizer未初始化，请设置LOCAL_MODEL_PATH或BASE_MODEL环境变量为本地模型路径")

    center_l = 1024
    center_r = 2048
    sigma_right_base = 600
    min_right_base = 0.2

    step = _get_dynamic_step(extra_info, "len")
    ratio = step / MAX_DYNAMIC_STEP if MAX_DYNAMIC_STEP > 0 else 1.0

    
    sigma_right = sigma_right_base * (1 - 0.5 * ratio)
    min_right = min_right_base * (1 - ratio)

    completions = [solution_str] if isinstance(solution_str, str) else solution_str

    rewards = []
    for content in completions:
        think = extract_think(content)

        if _effective_len_zero(think):
            rewards.append(0.0)
            print("len: 0 (effective) -> reward=0")
            continue

        gen_len = len(tok.encode(think))
        print("len:", gen_len)

        if gen_len <= center_l:
            reward = 1.0
        elif center_l < gen_len < center_r:
            reward = min_right + (1 - min_right) * math.exp(
                -((gen_len - center_l) ** 2) / (2 * sigma_right ** 2)
            )
        else:
            reward = 0.0

        rewards.append(reward)

    print("Final length rewards:", rewards)
    return rewards



def compute_score(data_source, solution_str, ground_truth, extra_info):
    res_f = format_reward(data_source, solution_str, ground_truth, extra_info)
    res_a = acc_reward(data_source, solution_str, ground_truth, extra_info)
    res_l = len_reward(data_source, solution_str, ground_truth, extra_info)

    total = (res_f[0] + res_a + res_l[0]) / 3.0

    return {
        "score": total,
        "format_score": res_f[0],
        "acc_score": res_a,
        "len_score": res_l[0],
    }
