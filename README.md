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
