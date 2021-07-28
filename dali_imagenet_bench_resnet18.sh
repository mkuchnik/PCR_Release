#!/bin/bash
NUM_GPUS=1
pcr_traindir="my_PCR_trainset"
valdir="valset"
cur_dir="$(pwd)"
script_name="${cur_dir}/dali_main.py"
batch_size=128
arch="resnet18"
lr=0.1
n_scans=$1
seed=${2:-0}

if [ "$#" -lt 1 ]; then
    echo "Requires number of scans"
fi

if [ $NUM_GPUS -gt 1 ]; then
  python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    ${script_name} \
    -a ${arch} --b ${batch_size} --lr=${lr}  \
    --PCR_train \
    --dali_cpu \
    --seed ${seed} \
    --n_scans=${n_scans} ${pcr_traindir} ${valdir}
else
  python3 ${script_name} \
    -a ${arch} --b ${batch_size} --lr=${lr}  \
    --PCR_train \
    --dali_cpu \
    --seed ${seed} \
    --n_scans=${n_scans} ${pcr_traindir} ${valdir}
fi
