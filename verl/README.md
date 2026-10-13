# WER training backend

This directory contains the modified verl backend for WER multi-phase GRPO
training. See the repository-root README and `scripts/train_wer.sh` for launch
instructions, data inputs, and task-executor configuration.

Upstream tests, unrelated training examples, standalone generation/evaluation
commands, and baseline reward implementations are excluded. The distributed
training, rollout, worker, and configuration components needed by the training
backend are retained.

Derived from [verl](https://github.com/volcengine/verl). The upstream Apache-2.0
license and copyright notice are retained in `LICENSE` and `Notice.txt`.
