#!/bin/bash

master_ip=$(ifconfig ens1 | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p')
master_port="8085"
echo $master_ip
pcr_traindir="my_PCR_trainset"
valdir="valset"
NUM_GPUS=1
WORLD_SIZE=1
cur_dir="$(pwd)"
script_name="${cur_dir}/dali_main.py"
batch_size=128
arch="resnet18"
#arch="shufflenet_v2_x1_0"
lr=0.1
seed=${2:-0}
n_scans=$1
host_prefix="h"
conda_env="py36env"

if [ "$#" -lt 1 ]; then
    echo "Requires number of scans"
fi

echo "Seed"
echo $seed

echo "Train"
echo $(ls $pcr_traindir | head)
echo "Val"
echo $(ls $valdir | head)

echo "Current Directory"
echo $cur_dir
last_n=$((${WORLD_SIZE}-1))
for i in `seq 1 ${last_n}`;
do
  cur_host=$host_prefix$i
  echo $cur_host
  ssh $cur_host\
    "screen -d -m \
    . ~/.bashrc  && \
    conda activate ${conda_env}  && \
    cd ${cur_dir} && \
    python -m torch.distributed.launch \
      --nproc_per_node=${NUM_GPUS} \
      --nnodes=${WORLD_SIZE} \
      --node_rank=$i \
      --master_addr="${master_ip}" \
      --master_port=${master_port} ${script_name} \
      -a ${arch} --b ${batch_size} --lr=${lr} \
      --PCR_train \
      --dali_cpu \
      --fp16 \
      --seed ${seed} \
      --n_scans=${n_scans} ${pcr_traindir} ${valdir}" &
done

echo "master"
. ~/.bashrc  && \
conda activate ${conda_env}  && \
python -m torch.distributed.launch \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${WORLD_SIZE} \
  --node_rank=0 \
  --master_addr=${master_ip} \
  --master_port=${master_port} ${script_name} \
  -a ${arch} --b ${batch_size} --lr=${lr}  \
  --PCR_train \
  --dali_cpu \
  --fp16 \
  --seed ${seed} \
  --n_scans=${n_scans} ${pcr_traindir} ${valdir}