# Overview

## Data preparation

* Training data are prepared following Kaldi voxceleb recipe, VoxCeleb2_dev and its four augmented versions are generated using RIRS_NOISES and MUSAN datasets.

* 40-dimensional (or 80-dimensional) FBANKs are extracted.

* ***Speaker augmentation is not applied***.

* ***Online data augmentation is not applied***.

## Models

* 1D-conv model: 512-d TDNN model following Kaldi voxceleb recipe

* 2D-conv models: Res2Net model and DPN (Dual Path Networks) model 

## Training

* Margin based (AAM-softmax, CM-softmax) loss functions.

* Scheduled learning rate and margin adjustments.

* Large margin finetuning (LMFT).

* Mixed precision training and distributed training.

## Scoring

* Cosine similarity scoring.

* Adaptive symmetric normalization (asnorm) on cosine scores.

* ***Quality measure function (QMF) is not applied***.

# Experiments

## Environment setup

* Install TensorFlow 1.x and Kaldi manually

* or setup a docker container with Dockerfile (recommended):

    ``docker build -t nvcr.io/nvidia/tensorflow:23.02-tf1-py3-kaldi .``
    
* or pull the container directly (recommended):
    
    ``docker pull xx205/ngc_tf1``

## Data generation

* Run docker container first: 

    ``docker run --rm -it --gpus device=all -v `pwd`:`pwd` --ipc=host --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:23.02-tf1-py3-kaldi``

* Clone this repository:

    ``git clone https://github.com/xx205/voxsrc2020_speaker_verification``

* Run data preparation script (40-dimensional FBANKs are extracted by default): 

    ``cd voxsrc2020_speaker_verification``
    
    ``bash prepare_data.sh``

## Model training

* Go to working directory: 

    ``cd tensorflow``

* Train a TDNN model with xx205's VoxSRC2020 setup on VoxCeleb2_dev_aug training set: 

    ``bash run_tdnn_local_voxsrc2020_vox2_dev_aug.sh``

* Export trained checkpoint to pb file for inference: 

    ``bash export_inference_model.sh models.tdnn_model tdnn exp/voxceleb2_dev_aug/tdnn_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs 2 40 cm_linear_voxsrc2020/kernel``

* Evaluate trained TDNN model performance on VoxCeleb1 Test/Extended/Hard trials: 

    ``bash eval_inference_model.sh exp/voxceleb2_dev_aug/tdnn_cm_linear_voxsrc2020_frames320_scale32_margin0.2_8GPUs_122636.pb 2``

* Train a Res2Net model with JTBD's VoxSRC2020 setup on VoxCeleb2_dev_aug training set:

    ``bash run_res2net_local_vox2_dev_aug.sh models.res2net_model res2net50_w24_s4_c64``

* Export trained checkpoint to pb file for inference:

    ``bash export_inference_model.sh models.res2net_model res2net50_w24_s4_c64 exp/voxceleb2_dev_aug/res2net50_w24_s4_c64_sc_cm_linear_frames200_scale32_margin0.2_8GPUs 3 40 sc_cm_linear/kernel``

* Evaluate trained Res2Net model performance on VoxCeleb1 Test/Extended/Hard trials:

    ``bash eval_inference_model.sh exp/voxceleb2_dev_aug/res2net50_w24_s4_c64_sc_cm_linear_frames200_scale32_margin0.2_8GPUs_122636.pb 3``

* Finetune the Res2Net model with LMFT on VoxCeleb2_dev training set:

    ``bash run_res2net_finetune_local_vox2_dev.sh models.res2net_model res2net50_w24_s4_c64``

* Export finetuned checkpoint to pb file for inference:

    ``bash export_inference_model.sh models.res2net_model res2net50_w24_s4_c64 exp/voxceleb2_dev/res2net50_w24_s4_c64_sc_cm_linear_frames600_scale32_margin0.4_8GPUs 3 40 sc_cm_linear/kernel``

* Evaluate finetuned Res2Net model performance on VoxCeleb1 Test/Extended/Hard trials:

    ``bash eval_inference_model.sh exp/voxceleb2_dev/res2net50_w24_s4_c64_sc_cm_linear_frames600_scale32_margin0.4_8GPUs_127968.pb 3``

# Results

## 1. VoxCeleb1_dev_aug as training data, 40-d FBANK features

### Model configurations
| model                     | loss function        | subcenter | sample length | margin     | # parameters | pooling function |
|---------------------------|----------------------|-----------|---------------|------------|--------------|------------------|
| tdnn_voxsrc2020           | cm_linear_voxsrc2020 | ✗         | 320           | (0.2, 0.1) | 3.5 M        | Stats Pool       |
| dpn68_voxsrc2020          | cm_linear_voxsrc2020 | ✗         | 320           | (0.2, 0.1) | 13.9 M       | Stats Pool       |

### Results on VoxCeleb1_Test
|                           | Vox1_Test (EER/minDCF0.01) | 
|---------------------------|----------------------------|
| tdnn_voxsrc2020 (cosine)  | 3.4398%/0.3339             |
| dpn68_voxsrc2020 (cosine) | 2.0894%/0.2544             |

## 2. VoxCeleb2_dev_aug as training data, 40-d FBANK features

### Model configurations
| model                           | loss function        | subcenter | sample length | margin     | # parameters | pooling function |
|---------------------------------|----------------------|-----------|---------------|------------|--------------|------------------|
| dpn68_voxsrc2020                | cm_linear_voxsrc2020 | ✗         | 320           | (0.2, 0.1) | 13.9 M       | Stats Pool       |
| res2net50_w24_s4_c64_voxsrc2020 | cm_linear_voxsrc2020 | ✗         | 320           | (0.2, 0.1) | 26.9 M       | Stats Pool       |

### Results on VoxCeleb1_Test/Extended/Hard
|                                          | Vox1_Test<br>(EER/minDCF0.01) | Extended<br>(EER/minDCF0.01) | Hard<br>(EER/minDCF0.01) |
|------------------------------------------|-------------------------------|------------------------------|--------------------------|
| dpn68_voxsrc2020 (cosine)                | 0.9517%/0.0884                | 1.0217%/0.1125               | 1.8649%/0.1833           |
| dpn68_voxsrc2020 (asnorm)                | 0.8347%/0.0879                | 0.9452%/0.0996               | 1.6401%/0.1527           |
| res2net50_w24_s4_c64_voxsrc2020 (cosine) | 0.9942%/0.1163                | 0.9924%/0.1152               | 1.8387%/0.1857           | 
| res2net50_w24_s4_c64_voxsrc2020 (asnorm) | 0.8400%/0.0931                | 0.9051%/0.0970               | 1.5976%/0.1471           |

## 3. VoxCeleb2_dev_aug as training data, 40-d FBANK features, LMFT on VoxCeleb2_dev

### Model configurations
| model                     | loss function | subcenter | sample length | margin     | # parameters | pooling function |
|---------------------------|---------------|-----------|---------------|------------|--------------|------------------|
| dpn68                     | sc_cm_linear  | ✓         | 200           | 0.2        | 13.9 M       | Stats Pool       |
| dpn68+LMFT                | sc_cm_linear  | ✓         | 600           | 0.4        | 13.9 M       | Stats Pool       |
| res2net50_w24_s4_c64      | sc_cm_linear  | ✓         | 200           | 0.2        | 26.9 M       | Stats Pool       |
| res2net50_w24_s4_c64+LMFT | sc_cm_linear  | ✓         | 600           | 0.4        | 26.9 M       | Stats Pool       |

### Results on VoxCeleb1_Test/Extended/Hard
|                                    | Vox1_Test<br>(EER/minDCF0.01) | Extended<br>(EER/minDCF0.01) | Hard<br>(EER/minDCF0.01) |
|------------------------------------|-------------------------------|------------------------------|--------------------------|
| dpn68 (cosine)                     | 1.0952%/0.1052                | 1.1725%/0.1322               | 2.0998%/0.2053           |
| dpn68 (asnorm)                     | 0.9783%/0.1081                | 1.0707%/0.1184               | 1.8656%/0.1742           |
| dpn68+LMFT (cosine)                | 0.8666%/0.0824                | 0.9634%/0.1058               | 1.6910%/0.1641           |
| dpn68+LMFT (asnorm)                | 0.8081%/0.0735                | 0.8934%/0.0925               | 1.5076%/0.1414           |
| res2net50_w24_s4_c64 (cosine)      | 1.0580%/0.1089                | 1.0728%/0.1226               | 1.9778%/0.1911           | 
| res2net50_w24_s4_c64 (asnorm)      | 0.9038%/0.0989                | 0.9824%/0.1086               | 1.7269%/0.1621           |
| res2net50_w24_s4_c64+LMFT (cosine) | 0.9198%/0.1005                | 0.9258%/0.1006               | 1.6957%/0.1650           | 
| res2net50_w24_s4_c64+LMFT (asnorm) | 0.7922%/0.0804                | 0.8482%/0.0903               | 1.4854%/0.1353           |

## 4. VoxCeleb2_dev_aug as training data, 80-d FBANK features, LMFT on VoxCeleb2_dev

### Model configurations
| model                          | loss function | subcenter | sample length | margin     | # parameters | pooling function |
|--------------------------------|---------------|-----------|---------------|------------|--------------|------------------|
| res2net50_w24_s4_c64           | sc_cm_linear  | ✓         | 200           | 0.2        | 32.2 M       | Stats Pool       |
| res2net50_w24_s4_c64+LMFT      | sc_cm_linear  | ✓         | 600           | 0.4        | 32.2 M       | Stats Pool       |
| res2net101_w24_s4_c32_att      | sc_cm_linear  | ✓         | 200           | 0.2        | 29.3 M       | Att Stats Pool   |
| res2net101_w24_s4_c32_att+LMFT | sc_cm_linear  | ✓         | 600           | 0.4        | 29.3 M       | Att Stats Pool   |
| res2net152_w24_s4_c32_att      | sc_cm_linear  | ✓         | 200           | 0.2        | 32.9 M       | Att Stats Pool   |
| res2net152_w24_s4_c32_att+LMFT | sc_cm_linear  | ✓         | 600           | 0.4        | 32.9 M       | Att Stats Pool   |
| res2net200_w24_s4_c32_att      | sc_cm_linear  | ✓         | 200           | 0.2        | 35.5 M       | Att Stats Pool   |
| res2net200_w24_s4_c32_att+LMFT | sc_cm_linear  | ✓         | 600           | 0.4        | 35.5 M       | Att Stats Pool   |

### Results on VoxCeleb1_Test/Extended/Hard, VoxSRC2022_dev
|                                         | Vox1_Test<br>(EER/minDCF0.01) | Extended<br>(EER/minDCF0.01) | Hard<br>(EER/minDCF0.01) | VoxSRC2022_dev<br>(EER/minDCF0.05) |
|-----------------------------------------|-------------------------------|------------------------------|--------------------------|------------------------------------|
| res2net50_w24_s4_c64 (cosine)           | 0.9304%/0.0855                | 1.0345%/0.1109               | 1.8228%/0.1777           | |
| res2net50_w24_s4_c64 (asnorm)           | 0.7656%/0.0662                | 0.8989%/0.0964               | 1.5381%/0.1423           | |
| res2net50_w24_s4_c64+LMFT (cosine)      | 0.7762%/0.0702                | 0.8589%/0.0933               | 1.5236%/0.1506           | |
| res2net50_w24_s4_c64+LMFT (asnorm)      | 0.6805%/0.0579                | 0.8024%/0.0827               | 1.3703%/0.1210           | |
| res2net101_w24_s4_c32_att (cosine)      | 0.6539%/0.0655                | 0.7837%/0.0823               | 1.4125%/0.1395           | |
| res2net101_w24_s4_c32_att (asnorm)      | 0.5742%/0.0664                | 0.7230%/0.0749               | 1.2679%/0.1187           | |
| res2net101_w24_s4_c32_att+LMFT (cosine) | 0.5795%/0.0512                | 0.6526%/0.0677               | 1.2099%/0.1174           | |
| res2net101_w24_s4_c32_att+LMFT (asnorm) | 0.5210%/0.0534                | 0.6081%/0.0618               | 1.0940%/0.0999           | |
| res2net152_w24_s4_c32_att (cosine)      | 0.5476%/0.0507                | 0.7313%/0.0811               | 1.3453%/0.1331           | |
| res2net152_w24_s4_c32_att (asnorm)      | 0.4891%/0.0530                | 0.6754%/0.0729               | 1.1957%/0.1101           | |
| res2net152_w24_s4_c32_att+LMFT (cosine) | 0.4732%/0.0425                | 0.6516%/0.0676               | 1.1957%/0.1119           | |
| res2net152_w24_s4_c32_att+LMFT (asnorm) | 0.4572%/0.0463                | 0.6099%/0.0587               | 1.0737%/0.0965           | |
| res2net200_w24_s4_c32_att (cosine)      | 0.4944%/0.0418                | 0.7137%/0.0780               | 1.2897%/0.1267           | |
| res2net200_w24_s4_c32_att (asnorm)      | 0.4200%/0.0501                | 0.6658%/0.0690               | 1.1496%/0.1019           | |
| res2net200_w24_s4_c32_att+LMFT (cosine) | 0.4041%/0.0390                | 0.6330%/0.0672               | 1.1641%/0.1113           | |
| res2net200_w24_s4_c32_att+LMFT (asnorm) | 0.3668%/0.0388                | 0.5930%/0.0581               | 1.0330%/0.0912           | 1.5017/0.0974 |