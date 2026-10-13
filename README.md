# WER Training Framework

This repository provides multi-phase GRPO training for the WER skill optimizer,
including task-based training rewards and data conversion between phases.

## Environment

Prepare an environment that meets the requirements of the original repositories:

- [verl](https://github.com/verl-project/verl)
- [Tau2-Bench](https://github.com/sierra-research/tau2-bench)
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

Follow their environment and dependency instructions for your training hardware.
Use the modified components included in this repository for WER training.

## Configuration

From the repository root, copy the configuration template:

```bash
cp env.example .env
```

Edit `.env` to set the local model path, training data, output location, hardware,
and task-executor endpoint. For Tau2 tasks, also fill in the user-simulator model
and credentials. Required settings and optional overrides are listed in
[env.example](env.example).

Load `.env` before starting training. Its `export` statements make the settings
available to the training script and Python processes:

```bash
source .env
```

## Models and training phases

All model paths and service model IDs are **placeholders**. Replace them in
`.env` before collecting data or training.

| Role | Configuration | Updated during training? |
| --- | --- | --- |
| Skill optimizer | `BASE_MODEL`: local optimizer checkpoint | Yes, continuously across all three phases |
| BFCL task executor | `WER_EXECUTOR_MODEL` and its endpoint | No |
| Tau2 task agent | `TAU2_AGENT_MODEL` and its endpoint; shares the executor service by default | No |
| Tau2 user simulator | `TAU2_USER_MODEL` and its endpoint | No |

The `openai/` prefix in Tau2 model IDs selects the OpenAI-compatible LiteLLM
interface. The model ID after the prefix must match the configured service.
Keep the task-executor and user-simulator services fixed throughout training.

| Step | Input data | Optimizer state |
| --- | --- | --- |
| Initial rollout collection | Existing seed skills and task IDs | No trainable optimizer is used |
| Parquet conversion | Seed-skill rollout JSONL | Uses the optimizer's tokenizer; makes no model API calls |
| Phase 0 | `TRAIN_FILE` | Initialized from `BASE_MODEL` |
| Phase 1 | `${WER_DATA_DIR}/1/train.parquet` | Continues with the weights learned in Phase 0 |
| Phase 2 | `${WER_DATA_DIR}/2/train.parquet` | Continues with the weights learned in Phase 1 |

The same executor, Tau2 agent, and user-simulator configuration is used for
initial rollouts and training rewards in every phase. The trainer does not load
a different optimizer checkpoint between phases. Initial seed-skill generation
is external to this snapshot, so it does not select a seed-generation model.
The default WER configuration uses programmatic task rewards and disables a
learned reward model, reference-model KL regularization, and a value critic.

## Training data

Prepare data in this order: **task execution with existing seed skills → rollout
JSONL → training parquet**. If you already have a training parquet, set
`TRAIN_FILE` in `.env` and skip the following steps. Model weights and initial
training data are not included.

### 1. Collect seed-skill rollouts

Starting from a task ID and its seed skill, run the task executor and record both
the execution trajectory and its task-success reward. The collection functions
below are Python library functions; use them in your own collection driver to
write the initial JSONL.

For **BFCL**:

1. Call `run_single_task_inference()` in
   [task_execution.py](verl/berkeley-function-call-leaderboard/bfcl_eval/training/task_execution.py)
   with `run_id`, `skill`, `result_dir`, `sequence_1="origin"`, and a trial label
   such as `sequence_2="traj0"`. Repeat with `sequence_2="traj1"` to collect two
   executions for the same task and skill.
2. Call `evaluate_single_task_result()` in
   [task_reward.py](verl/berkeley-function-call-leaderboard/bfcl_eval/training/task_reward.py)
   with the matching task ID, sequence labels, and result directory to obtain
   each execution's reward. Use the returned inference result's `inference_log`
   as that execution's `trajectory`.

For **Tau2**:

- Call `evaluate_tau2_task()` in
  [task_reward.py](verl/tau2-bench-verified/src/tau2/training/task_reward.py)
  with `domain`, `task_id`, and `skill`. Pass the configured agent/user models and
  API settings through `agent_llm`, `user_llm`, `agent_llm_args`, and
  `user_llm_args` when overriding the `.env` defaults. The function runs exactly
  two trials and returns their rewards and serialized trajectories in
  `result.trials`.

Save the records to `${WER_DATA_DIR}/origin/seed_skill_rollouts.jsonl`, with one
JSON object per line and one record per task:

| Field | Content |
| --- | --- |
| `run_id` | Task ID matching the bundled initial-context data |
| `skill` | Seed skill used by the task executor |
| `reward` | Sum of the two execution rewards |
| `trajectories` | Two objects, each containing `reward` and `trajectory` |
| `sequence_1` | BFCL only: use `"origin"` for initial records |
| `domain` | Tau2 only: `airline`, `retail`, or `telecom` |

BFCL trajectory objects can also retain their `sequence_2` trial labels. Omit
`domain` from BFCL records: the converter uses its presence to identify Tau2
records. BFCL task IDs must be unique; Tau2 `(domain, run_id)` pairs must be
unique. Both types can be combined in the same JSONL.

The training reward code in
[skill.py](verl/berkeley-function-call-leaderboard/bfcl_eval/skill.py), specifically
`_acc_reward_bfcl()` and `_acc_reward_tau2()`, shows how execution results are
assembled into this record format. Use it as a reference for the initial
collection driver.

### 2. Convert rollouts to training parquet

With `.env` loaded, run from the repository root:

```bash
PYTHONPATH="$PWD/verl${PYTHONPATH:+:$PYTHONPATH}" \
python -m verl.datapreprocess.build_seed_skill_train_parquet \
  --rollouts-path "$WER_DATA_DIR/origin/seed_skill_rollouts.jsonl" \
  --output-train-file "$TRAIN_FILE" \
  --tokenizer-model-path "$BASE_MODEL"
```

The entry point is
[build_seed_skill_train_parquet.py](verl/verl/datapreprocess/build_seed_skill_train_parquet.py).
It reads the rollout JSONL, matches task IDs to the bundled initial contexts,
builds optimization prompts, and writes the training parquet. Context paths can
be overridden in `.env`. Prompt construction and trajectory processing are in
[skill4bfcl4phase.py](verl/verl/datapreprocess/skill4bfcl4phase.py).

The initial converter drops prompts longer than 18,000 tokens. It retains up to
13 BFCL records per task class across task IDs `multi_turn_base_0` through
`multi_turn_base_199`, dropping BFCL IDs outside those four classes. Tau2 records
are not subject to this class cap.

The output parquet contains `data_source`, `prompt`, `ability`, `reward_model`,
and `extra_info`; its schema is defined by `build_train_features()` in the
converter. After training starts, the reward module saves phase records and the
trainer automatically converts them into the next phase's parquet.

## Start training

With the configuration loaded and the task-executor service available, run:

```bash
bash scripts/train_wer.sh
```

The default configuration uses six Ascend NPUs, GRPO, and three training phases
with two epochs each. Training settings are defined in
[scripts/train_wer.sh](scripts/train_wer.sh). Additional Hydra overrides can be
passed to the launcher:

```bash
bash scripts/train_wer.sh actor_rollout_ref.actor.optim.lr=1e-6
```

Logs, rollout records, checkpoints, and task-reward artifacts are written under
`OUTPUT_DIR`. Phase records and generated training parquet files are written
under `WER_DATA_DIR`; the trainer prepares the next phase's data automatically.

Multi-phase training currently starts from scratch and does not support
checkpoint resume. Use a fresh output directory for each run, or pass
`trainer.resume_mode=disable` to start over in an existing directory. Phase
conversion requires at least one retained skill/trajectory record from the
previous phase; an empty retained dataset stops the transition.

## Licenses

Original WER code is covered by the root MIT license. The included verl, BFCL,
and Tau2 components retain their respective licenses and copyright notices.
