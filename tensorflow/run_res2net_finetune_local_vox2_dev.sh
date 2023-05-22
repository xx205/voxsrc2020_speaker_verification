#!/bin/sh

. ../global_config.sh

model_source=${1:-"models.res2net_model"}
model_id=${2:-'res2net50_w24_s4_c64'}
expansion_dim=${3:-3}
projection_id=${4:-'sc_cm_linear'}
scale=${5:-32}
margin=${6:-0.2}
num_shards_per_rank=${7:-4}
batch_size=${8:-32}
num_accumulation_steps=${9:-4}
total_epochs=23

# dataset description
dataset=voxceleb2_dev_aug
dataset_length=$((1092009 * 5))
num_classes=5994
frames=200
# feat_dim=80

# num_gpus=`nvidia-smi -L | wc -l`

pretrain_exp_dir="exp/${dataset}/${model_id}_${projection_id}_frames${frames}_scale${scale}_margin${margin}_${num_gpus}GPUs"

# finetune model with
dataset=voxceleb2_dev
batch_size=16
num_accumulation_steps=8
margin=0.4
frames=600
total_epochs=24

exp_dir="exp/${dataset}/${model_id}_${projection_id}_frames${frames}_scale${scale}_margin${margin}_${num_gpus}GPUs"
mkdir -p `dirname ${exp_dir}`
ln -s `pwd`/${pretrain_exp_dir} ${exp_dir}


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
