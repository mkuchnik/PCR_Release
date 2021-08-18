data_dir="/dev/null"
# Point to directory with datasets
dataset_base_dir="$HOME/datasets"
posix_throttle_token_rate=0 # kiBps
model="resnet18"
model="shufflenet_v2_x1_0"
scan=0 # NOTE change below
image_size=160
batch_size=128
epochs=90
dataset_metadata=""
prefix_dir="."
master_ip=$(ifconfig ens1 | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p')
echo $master_ip
master_port="8085"
NUM_GPUS=1
WORLD_SIZE=1
cur_dir="$(pwd)"
global_settings=""
script_name=$(pwd)/main.py
env_str=""
seed=0
# NOTE: We use the prefix 'h' for hosts (e.g., h0, h1, h2, ...).
# Change this if your setup uses a different prefix

if [ "$posix_throttle_token_rate" -ne "0" ]
then
  env_str+=" export POSIX_THROTTLE_TOKEN_RATE=${posix_throttle_token_rate} && "
else
  env_str+=" echo 'no_throttle' && "
fi

kill_gpu_tasks() {
  # TODO(mkuchnik): This throws weird errors sometimes
  last_n=$((${WORLD_SIZE}-1))
  pdsh -w h[0-$last_n] "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 kill -9"
  # NOTE: may have to perform kill on 'ps o ppid'
  cmd="kill -9 \$(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')"
  pdsh -w h[0-23] "${cmd}"
  cmd="killall -9 python"
  pdsh -w h[0-23] "${cmd}"
  cmd="kill -9 \$(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')"
  pdsh -w h-[0-23] "${cmd}"
  cmd="killall -9 python"
  pdsh -w h-[0-23] "${cmd}"

  nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -r -n1 echo
  nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -r -n1 kill -9
  nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -r -n1 ps o ppid | xargs -n1 echo
  pdsh -w h[0-23] "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 kill -9"
  pdsh -w h[0-23] "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 ps o ppid | xargs -r -n1 kill -9"
  sudo fuser -v /dev/nvidia*
  for i in $(sudo lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done
  pdsh -w h[0-26] "sudo sync && sudo /sbin/sysctl vm.drop_caches=3 && sudo sync"
}

run_tfdata_training() {
  kill_gpu_tasks
  checkpoint_prefix="$prefix_dir/checkpoints_$scan"
  echo "Current Directory"
  echo $cur_dir
  echo "Script $script_name"
  last_n=$((${WORLD_SIZE}-1))
  for i in `seq 1 ${last_n}`;
  do
    echo h$i
    ssh h$i\
      "screen -d -m \
      . ~/.bashrc  && \
      conda activate py37  && \
      eval ${env_str} \
      cd ${cur_dir} && \
      python -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS} \
        --nnodes=${WORLD_SIZE} \
        --node_rank=${i} \
        --master_addr="${master_ip}" \
        --master_port=${master_port} \
        ${script_name} \
        $data_dir --train_loader=tensorflow \
        -a $model -b $batch_size --scan=$scan $global_settings \
        --checkpoint_prefix=$checkpoint_prefix --local_rank=$i \
        --world-size=${WORLD_SIZE} \
        --dist-url=\"tcp://${master_ip}:${master_port}\"" &
  done
  echo "master"
  . ~/.bashrc  && \
  conda activate py37  && \
  eval ${env_str} \
  python -m torch.distributed.launch \
      --nproc_per_node=${NUM_GPUS} \
      --nnodes=${WORLD_SIZE} \
      --node_rank=0 \
      --master_addr="${master_ip}" \
      --master_port=${master_port} \
      ${script_name} \
      $data_dir --train_loader=tensorflow \
      -a $model -b $batch_size --scan=$scan $global_settings \
      --checkpoint_prefix=$checkpoint_prefix --local_rank=0 \
      --world-size=${WORLD_SIZE} \
      --dist-url="tcp://${master_ip}:${master_port}"
}

run_imagenet_tfr_training() {
  data_dir="$dataset_base_dir/ImageNet/ILSVRC12_TFRecord/train/train\*"
  val_data_dir="$dataset_base_dir/ImageNet/ILSVRC12_TFRecord/validation/val\*"
  data_format="TFRecord"
  val_data_format="TFRecord"
  validation_loader="tensorflow"
  global_settings="--validate-freq=1 --checkpoint-freq=1 --data_format=$data_format --lr=0.1 --image_size=$image_size --val_data_dir=$val_data_dir --imagenet_training --scale_lr --autotune-freq=0 --val_data_format=$val_data_format --validation_loader=$validation_loader --epochs=$epochs"
  run_tfdata_training
}

run_imagenet_pcr_training() {
  data_dir="$dataset_base_dir/ImageNet/ILSVRC12_PCR/train/\*.pcr"
  val_data_dir="$dataset_base_dir/ImageNet/ILSVRC12_TFRecord/validation/val\*"
  data_format="PCR"
  val_data_format="TFRecord"
  validation_loader="tensorflow"
  global_settings="--validate-freq=1 --checkpoint-freq=1 --data_format=$data_format --lr=0.1 --image_size=$image_size --val_data_dir=$val_data_dir --imagenet_training --scale_lr --autotune-freq=0 --val_data_format=$val_data_format --validation_loader=$validation_loader --epochs=$epochs"
  run_tfdata_training
}

run_imagenet_pcr_training_autotune() {
  data_dir="$dataset_base_dir/ImageNet/ILSVRC12_PCR/train/\*.pcr"
  val_data_dir="$dataset_base_dir/ImageNet/ILSVRC12_TFRecord/validation/val\*"
  data_format="PCR"
  val_data_format="TFRecord"
  validation_loader="tensorflow"
  global_settings="--validate-freq=1 --checkpoint-freq=1 --data_format=$data_format --lr=0.1 --image_size=$image_size --val_data_dir=$val_data_dir --imagenet_training --scale_lr --autotune-freq=15 --val_data_format=$val_data_format --validation_loader=$validation_loader --epochs=$epochs"
  run_tfdata_training
}

for trial in 0;
do
  dataset_metadata="test_run_tfr"
  for scan in 0;
  do
    seed=$trial
    prefix_dir="${dataset_metadata}/scan_$scan/trial_$trial"
    mkdir -p $prefix_dir
    run_imagenet_tfr_training | tee $prefix_dir/log_$scan.txt
  done
done

for trial in 0;
do
  dataset_metadata="test_run_pcr"
  for scan in 1 2 5 10;
  do
    seed=$trial
    prefix_dir="${dataset_metadata}/scan_$scan/trial_$trial"
    mkdir -p $prefix_dir
    run_imagenet_pcr_training | tee $prefix_dir/log_$scan.txt
  done
done

for trial in 0;
do
  dataset_metadata="test_run_autotune"
  for scan in 10;
  do
    seed=$trial
    prefix_dir="${dataset_metadata}/scan_$scan/trial_$trial"
    mkdir -p $prefix_dir
    run_imagenet_pcr_training_autotune | tee $prefix_dir/log_$scan.txt
  done
done