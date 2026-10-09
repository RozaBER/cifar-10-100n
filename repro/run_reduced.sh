#!/usr/bin/env bash
# Reduced-budget CE reproduction for a CPU-only machine:
# ResNet-18, 10 epochs (lr 0.1 for 6 epochs, then 0.01), human labels.
# The paper setting is ResNet-34 for 100 epochs: drop --model/--n_epoch below to run it on a GPU.
# Re-running the script resumes unfinished runs from their checkpoints and skips finished ones.
set -u
cd "$(dirname "$0")/.."
mkdir -p repro/logs repro/ckpt

N_EPOCH=${N_EPOCH:-10}
MODEL=${MODEL:-ResNet18}
EXTRA=${EXTRA:---channels_last --num_workers 1}
RUNS=${RUNS:-"cifar10:clean cifar10:worst cifar10:aggre cifar100:noisy100"}

for run in $RUNS; do
    dataset=${run%%:*}
    noise=${run##*:}
    name="${dataset}_${noise}_human_${MODEL}_e${N_EPOCH}"
    log="repro/logs/${name}.log"
    if grep -q '^final:' "$log" 2>/dev/null; then
        echo "skip $name (done)"
        continue
    fi
    echo "=== $(date '+%F %T') start $name" | tee -a "$log"
    python3 main.py --dataset "$dataset" --noise_type "$noise" --is_human \
        --model "$MODEL" --n_epoch "$N_EPOCH" --ckpt "repro/ckpt/${name}.pt" $EXTRA >> "$log" 2>&1
    echo "=== $(date '+%F %T') exit $? $name" | tee -a "$log"
done
