#!/bin/bash
# Copyright (c) 2020 Xu Xiang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -xe

. ../global_config.sh

hosts=localhost
# num_gpus=8
# Number of GPUs for training and evaluation
# num_gpus=`nvidia-smi -L | wc -l`

uuid=`python3 -c 'import uuid; print(uuid.uuid4())'`
hostfile=/tmp/hostfile-${uuid}
cat > ${hostfile} <<EOF
EOF

np=0
for host in $hosts
do
    echo $host "slots=$num_gpus" >> ${hostfile}
    np=$(($np + $num_gpus))
done

# data_dir="${np}-gpu"

i=0
echo "running..."

export OMPI_MCA_btl_openib_allow_ib=1

export TF_GPU_ALLOCATOR=cuda_malloc_async

py_main=$1
module_source=$2
model_id=$3
expansion_dim=$4
batch_size=$5
dataset_length=$6
feat_rspec_tpl=$7
utt2id_pkl=$8
utt2kwd_pkl=$9
cmvn_pkl=${10}
num_classes=${11}
projection=${12}
scale=${13}
margin=${14}
feat_dim=${15}
feat_length=${16}
min_feat_length=${17}
max_feat_length=${18}
training=${19}
specaug=${20}
num_shards_per_rank=${21:-1}
num_accumulation_steps=${22:-1}
total_epochs=${23:-1}
exp_dir=${24}

mpirun --allow-run-as-root \
       --oversubscribe -np ${np} -hostfile ${hostfile} -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x TF_GPU_ALLOCATOR=cuda_malloc_async \
       -mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
       python3 -u $py_main \
       --module_source $module_source \
       --model_id $model_id \
       --expansion_dim $expansion_dim \
       --batch_size $batch_size \
       --dataset_length $dataset_length \
       --feat_rspec_tpl "$feat_rspec_tpl" \
       --utt2id_pkl "$utt2id_pkl" \
       --utt2kwd_pkl "$utt2kwd_pkl" \
       --cmvn_pkl "$cmvn_pkl" \
       --num_classes $num_classes \
       --projection $projection \
       --scale $scale \
       --margin $margin \
       --feat_dim $feat_dim \
       --feat_length $feat_length \
       --min_feat_length $min_feat_length \
       --max_feat_length $max_feat_length \
       --training $training \
       --specaug $specaug \
       --num_shards_per_rank ${num_shards_per_rank} \
       --num_accumulation_steps $num_accumulation_steps \
       --total_epochs $total_epochs \
       --exp_dir $exp_dir \
       --uuid $uuid
wait
