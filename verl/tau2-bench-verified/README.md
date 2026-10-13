# Tau2 training reward runtime

This directory contains the task environments, agents, user simulator, and
single-task reward helper required for WER training with Tau2 records.

`src/tau2/training/task_reward.py` is called by the WER reward module. Standalone
benchmark commands, baseline experiments, task-generation scripts, and result
analysis are removed. See the repository-root README for training configuration.

Derived from [Tau2](https://github.com/sierra-research/tau2-bench).
The upstream MIT license is retained in `LICENSE`.
