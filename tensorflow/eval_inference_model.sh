#!/bin/bash
# Copyright (c) 2023 Xu Xiang
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


. ../global_config.sh

model=${1:-"exp/voxceleb2_dev/tdnn_length320_cm_linear_voxsrc2020_scale32.0_margin0.2_8GPUs_122636.pb"}
expansion_dim=${2:-2}

# Number of GPUs for training and evaluation
# num_gpus=`nvidia-smi -L | wc -l`

dir=`pwd`/${model%.pb}_embeddings

for dataset in voxceleb2_dev voxceleb1; do
    mkdir -p ${dir}/${dataset}
    for i in `seq 1 ${num_gpus}`; do
		export CUDA_VISIBLE_DEVICES=$(($i - 1))
		python3 tf_extract.py \
			--pb-file ${model} \
			--expand-dim ${expansion_dim} \
			--rspec `pwd`/../data/${dataset}/${num_gpus}-split/feats.${i} \
			--wspec ${dir}/${dataset}/xvector.${i} &
    done
    wait
    cat `eval echo ${dir}/${dataset}/xvector.{1..${num_gpus}}.ark` \
	    > ${dir}/${dataset}/xvector.ark
done

for testset in T E H; do
    python3 snorm.py \
	    --trial ../data/voxceleb1_trials/list_test_${testset}.txt \
	    --test_ark ${dir}/voxceleb1/xvector.ark \
	    --cosine_score ${dir}/voxceleb1/cosine_${testset}.txt \
	    --cohort_ark ${dir}/voxceleb2_dev/xvector.ark \
	    --cohort_spk2utt ../data/voxceleb2_dev/spk2utt \
	    --snorm_score ${dir}/voxceleb1/snorm_${testset}.txt &
done
wait

for testset in T E H; do
    python3 eer_minDCF.py \
	    --trial ../data/voxceleb1_trials/list_test_${testset}.txt \
	    --score ${dir}/voxceleb1/cosine_${testset}.txt
    python3 eer_minDCF.py \
	    --trial ../data/voxceleb1_trials/list_test_${testset}.txt \
	    --score ${dir}/voxceleb1/snorm_${testset}.txt
done
