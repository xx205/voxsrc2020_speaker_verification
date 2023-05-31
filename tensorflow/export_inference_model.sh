#!/bin/bash
# Copyright (c) 2020-2023 Xu Xiang
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

# set -xe
module_source=${1:-"models.tdnn_model"}
model_id=${2:-"tdnn"}
saved_directory=${3:-"exp/voxceleb2_dev/tdnn_length320_cm_linear_voxsrc2020_scale32.0_margin0.2_8GPUs"}
# expansion_dim is 2 for 1D conv model (e.g. TDNN), 3 for 2D conv model (e.g. Res2Net)
expansion_dim=${4:-2}
# feat_dim=${5:-80}
projection_name=${5:-"cm_linear_voxsrc2020/kernel"}


# export inference graph
python3 export_inference_graph.py --module_source $module_source \
    --model_id $model_id \
    --checkpoint_directory $saved_directory \
    --expansion_dim $expansion_dim \
    --feat_dim $feat_dim

model_checkpoint_path=`head -n 1 ${saved_directory}/checkpoint | awk '{print $2}' | sed 's#"##g'`
model_checkpoint_iter=`echo $model_checkpoint_path | awk -F- '{print $2}'`

# export inference pb file
freeze_graph --input_graph=${saved_directory}/graph_eval.pbtxt \
    --input_binary=false \
    --output_node_names=outputs \
    --input_checkpoint=${saved_directory}/${model_checkpoint_path} \
    --output_graph=${saved_directory}"_${model_checkpoint_iter}".pb

# export weight matrix and use weight columns as cohort speaker embeddings
python3 export_projection_weight.py --input_checkpoint ${saved_directory}/${model_checkpoint_path} \
    --var_name $projection_name \
    --output_weight ${saved_directory}"_${model_checkpoint_iter}".pkl
