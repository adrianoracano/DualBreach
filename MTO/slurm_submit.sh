#!/bin/bash
# usage: bash slurm_submit.sh <name> [extra args for cold_decoding.py, e.g. --pretrained_model Qwen3.5-4b --guard_loss margin]
# Submits one MTO attack (sample 0 of advbench_original by default) as a 2-GPU Slurm job: inside the job cuda:0 is the target
# LLM (+ lazily loaded response model) and cuda:1 the d3 guard (+ lazily loaded judge). Override the sample/dataset with
# START, END (inclusive) and DATASET, and the time limit with TIME (default 02:00:00). Logs go to outputs/slurm/<name>-<jobid>.log; results are tagged with `--run_tag _<name>`.
name=$1; shift
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p outputs/slurm
sbatch --job-name="$name" --gres=gpu:2 --cpus-per-task=8 --mem=64G --time=${TIME:-02:00:00} \
    --output="outputs/slurm/%x-%j.log" <<EOS
#!/bin/bash
cd $PWD
source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash guard_attack.sh suffix ${START:-0} 0 1 ${DATASET:-advbench_original} --guard --end ${END:-0} --run_tag _$name \\
    --response_device cuda:0 --judge_device cuda:1 $(printf '%q ' "$@")
EOS
