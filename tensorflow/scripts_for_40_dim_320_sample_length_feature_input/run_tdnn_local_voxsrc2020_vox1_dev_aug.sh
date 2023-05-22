#!/bin/sh

. ../global_config.sh

model_source=${1:-"models.tdnn_model"}
model_id=${2:-'tdnn'}
expansion_dim=${3:-2}
projection_id=${4:-'cm_linear_voxsrc2020'}
scale=${5:-32}
margin=${6:-0.2}
num_shards_per_rank=${7:-4}
batch_size=${8:-128}
num_accumulation_steps=${9:-1}
total_epochs=23


# dataset description
dataset=voxceleb1_dev_aug
dataset_length=$((148642 * 5))
num_classes=1211
frames=320
# feat_dim=40


# num_gpus=`nvidia-smi -L | wc -l`

exp_dir="exp/${dataset}/${model_id}_${projection_id}_frames${frames}_scale${scale}_margin${margin}_${num_gpus}GPUs"
mkdir -p ${exp_dir}

bash ./run_tf_train_local.sh tf_train_tdnn.py \
    ${model_source} \
    ${model_id} \
    ${expansion_dim} \
    ${batch_size} \
    ${dataset_length} \
    "apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:../data/${dataset}/{} ark:-|" \
    "../data/${dataset}/utt2id.pkl" \
    'None' \
    'None' \
    ${num_classes} \
    ${projection_id} \
    ${scale} \
    ${margin} \
    ${feat_dim} \
    ${frames} \
    160 \
    10000 \
    True \
    False \
    ${num_shards_per_rank} \
    ${num_accumulation_steps} \
    ${total_epochs} \
    ${exp_dir}
