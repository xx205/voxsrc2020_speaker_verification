# Environment setup

* Software requirements:

    Install TensorFlow 1.x and Kaldi manually or setup a docker container with ``Dockerfile``:

    ``docker build -t nvcr.io/nvidia/tensorflow:23.02-tf1-py3-kaldi .``

    or pull the container directly:

    ``docker pull xx205/ngc_tf1``

* Hardware requirements:

    NVIDIA GPUs for training models, 1 TB SSD (or HDD) for data storage


# Data preparation

* Run the docker container first: ``docker run --rm -it --gpus device=all -v `pwd`:`pwd` --ipc=host --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:23.02-tf1-py3-kaldi``
* Run the data preparation script: ``bash prepare_data.sh``


# Runing experiments

* ``cd tensorflow``
* Train a TDNN model with VoxSRC 2020 competition setup on VoxCeleb2 dev training set (without data augmentation): ``bash run_tdnn_local_voxsrc2020_vox2_dev.sh``
* Export trained checkpoint to pb file for inference: ``bash export_inference_model.sh models.tdnn_model tdnn exp/voxceleb2_dev/tdnn_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs 2 40 cm_linear_voxsrc2020/kernel``
* Evaluate model performance on VoxCeleb1 Test/Extended/Hard trials: ``bash eval_inference_model.sh exp/voxceleb2_dev/tdnn_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs_122636.pb 2 40``

# Experiment results with VoxSRC 2020 Team xx205's settings

* Train on VoxCeleb1 dev, evaluate on VoxCeleb1 Test, cosine scoring 
    |                                                                               | Test           | num params |
    |-------------------------------------------------------------------------------|----------------|------------|
    | dpn68_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs_16675_embeddings | 2.0894%/0.2544 | 13925494 (13.9M)   |
    | tdnn_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs_16675_embeddings  | 3.4398%/0.3339 | 3524096 (3.5M)   |

* Train on VoxCeleb2 dev, evaluate on VoxCeleb1 Test / Extended / Hard
    |                                                                                               | Test           | Extended       | Hard           | num params       |
    |-----------------------------------------------------------------------------------------------|----------------|----------------|----------------|------------------|
    | dpn68_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs_16675_embeddings                 | 0.9517%/0.0884 | 1.0217%/0.1125 | 1.8649%/0.1833 | 13925494 (13.9M) |
    | + snorm                                                                                       | 0.8347%/0.0879 | 0.9452%/0.0996 | 1.6401%/0.1527 |                  |
    | res2net50_w24_s4_c64_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs_122636_embeddings | 0.9942%/0.1163 | 0.9924%/0.1152 | 1.8387%/0.1857 | 26925168 (26.9M) |
    | + snorm                                                                                       | 0.8400%/0.0931 | 0.9051%/0.0970 | 1.5976%/0.1471 |                  |

# Experiment results with VoxSRC 2020 Team JTBD's settings

* Train on VoxCeleb2 dev, evaluate on VoxCeleb1 Test / Extended / Hard
    |                                                                                              | Test           | Extended       | Hard           | num params       |
    |----------------------------------------------------------------------------------------------|----------------|----------------|----------------|------------------|
    | dpn68_sc_cm_linear_frames200_scale32_margin0.2_8GPUs_122636_embeddings                       | 1.0952%/0.1052 | 1.1725%/0.1322 | 2.0998%/0.2053 | 13925494 (13M)   |
    | + snorm                                                                                      | 0.9783%/0.1081 | 1.0707%/0.1184 | 1.8656%/0.1742 |                  |
    | dpn68_sc_cm_linear_frames600_scale32_margin0.4_8GPUs_127968_embeddings (LMFT)                | 0.8666%/0.0824 | 0.9634%/0.1058 | 1.6910%/0.1641 |                  |
    | + snorm                                                                                      | 0.8081%/0.0735 | 0.8934%/0.0925 | 1.5076%/0.1414 |                  |
    | res2net50_w24_s4_c64_sc_cm_linear_frames200_scale32_margin0.2_8GPUs_122636_embeddings        | 1.0580%/0.1089 | 1.0728%/0.1226 | 1.9778%/0.1911 | 26925168 (26.9M) |
    | + snorm                                                                                      | 0.9038%/0.0989 | 0.9824%/0.1086 | 1.7269%/0.1621 |                  |
    | res2net50_w24_s4_c64_sc_cm_linear_frames600_scale32_margin0.4_8GPUs_127968_embeddings (LMFT) | 0.9198%/0.1005 | 0.9258%/0.1006 | 1.6957%/0.1650 |                  |
    | + snorm                                                                                      | 0.7922%/0.0804 | 0.8482%/0.0903 | 1.4854%/0.1353 |                  |



# Experiment results with VoxSRC 2020 Team JTBD's settings and 80-dimensional FBANK input features

* Train on VoxCeleb2 dev, evaluate on VoxCeleb1 Test / Extended / Hard
    |                                                                                                   | Test           | Extended       | Hard           | num params       |
    |---------------------------------------------------------------------------------------------------|----------------|----------------|----------------|------------------|
    | res2net50_w24_s4_c64_sc_cm_linear_frames200_scale32_margin0.2_8GPUs_122636_embeddings             | 0.9304%/0.0855 | 1.0345%/0.1109 | 1.8228%/0.1777 | 32209008 (32.2M) |
    | + snorm                                                                                           | 0.7656%/0.0662 | 0.8989%/0.0964 | 1.5381%/0.1423 |                  |
    | res2net50_w24_s4_c64_sc_cm_linear_frames600_scale32_margin0.4_8GPUs_127968_embeddings (LMFT)      | 0.7762%/0.0702 | 0.8589%/0.0933 | 1.5236%/0.1506 |                  |
    | + snorm                                                                                           | 0.6805%/0.0579 | 0.8024%/0.0827 | 1.3703%/0.1210 |                  |
    | res2net101_w24_s4_c32_att_sc_cm_linear_frames200_scale32_margin0.2_8GPUs_122636_embeddings        | 0.6539%/0.0655 | 0.7837%/0.0823 | 1.4125%/0.1395 | 29289040 (29.3M) |
    | + snorm                                                                                           | 0.5742%/0.0664 | 0.7230%/0.0749 | 1.2679%/0.1187 |                  |
    | res2net101_w24_s4_c32_att_sc_cm_linear_frames600_scale32_margin0.4_8GPUs_127968_embeddings (LMFT) | 0.5795%/0.0512 | 0.6526%/0.0677 | 1.2099%/0.1174 |                  |
    | + snorm                                                                                           | 0.5210%/0.0534 | 0.6081%/0.0618 | 1.0940%/0.0999 |                  |
