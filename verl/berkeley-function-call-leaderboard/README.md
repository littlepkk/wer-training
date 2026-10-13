# BFCL training reward runtime

This directory contains the BFCL multi-turn task environments, local executor
handler, and task-success checker required by WER training. The retained BFCL
data is the multi-turn base task set and its training-context compilation.

The task helpers in `bfcl_eval/training/` are called by `bfcl_eval/skill.py`.
Standalone leaderboard evaluation and multi-model benchmark commands are removed.
See the repository-root README for training configuration.

Derived from [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).
The upstream Apache-2.0 license is retained in `LICENSE`.
