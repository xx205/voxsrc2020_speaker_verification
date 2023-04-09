#!/bin/bash

model=${1:-"tdnn_length320_cm_linear_voxsrc2020_scale32.0_margin0.2_8GPUs_0.pb"}
expansion_dim=${2:-2}
fbank_dim=40

# Number of GPUs for training and evaluation
num_gpus=`nvidia-smi -L | wc -l`

dir=`pwd`/${model%.pb}_embeddings

# for dataset in voxceleb2_dev voxceleb1; do
for dataset in voxceleb1; do
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

# for testset in T E H; do
for testset in T; do
    python3 snorm.py \
	    --trial ../data/voxceleb1_trials/list_test_${testset}.txt \
	    --test_ark ${dir}/voxceleb1/xvector.ark \
	    --cosine_score ${dir}/voxceleb1/cosine_${testset}.txt \
	    --cohort_ark ${dir}/voxceleb2_dev/xvector.ark \
	    --cohort_spk2utt ../data/voxceleb2_dev/spk2utt \
	    --snorm_score ${dir}/voxceleb1/snorm_${testset}.txt &
done
wait

# for testset in T E H; do
for testset in T; do
    python3 eer_minDCF.py \
	    --trial ../data/voxceleb1_trials/list_test_${testset}.txt \
	    --score ${dir}/voxceleb1/cosine_${testset}.txt
    python3 eer_minDCF.py \
	    --trial ../data/voxceleb1_trials/list_test_${testset}.txt \
	    --score ${dir}/voxceleb1/snorm_${testset}.txt
done
