# ST-GCN
An **unofficial** Tensorflow implementation of the paper "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" in AAAI 2018.

- Paper: [PDF](https://arxiv.org/abs/1801.07455)
- Code is based on [mmskeleton](https://github.com/open-mmlab/mmskeleton) and [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)

# Model Weights
Model weights for ST-GCN trained on xview and xsub joint data [Dropbox](https://www.dropbox.com/sh/cuq10z9w1ztqpuj/AACgJOgoG5AOw3-aBHAWWO5sa?dl=0)

## Dependencies

- Python >= 3.5
- scipy >= 1.3.0
- numpy >= 1.16.4
- tensorflow >= 2.0.0

## Directory Structure

Most of the interesting stuff can be found in:
- `model/stgcn.py`: model definition of ST-GCN
- `data_gen/`: how raw datasets are processed into numpy tensors
- `graphs/ntu_rgb_d.py`: graph definition
- `main.py`: general training/eval processes; etc.

## Downloading & Generating Data

### NTU RGB+D

1. The [NTU RGB+D dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf) can be downloaded from [here](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp). We'll only need the Skeleton data (~ 5.8G).

2. After downloading, unzip it and put the folder `nturgb+d_skeletons` to `./data/nturgbd_raw/`.

3. Generate the joint dataset first:

```bash
cd data_gen
python3 gen_joint_data.py
```

Specify the data location if the raw skeletons data are placed somewhere else. The default looks at `./data/nturgbd_raw/`.

4. Generate the tfrecord files for joint data :

```bash
python3 gen_tfrecord_data.py
```

## Training

To start training the network with the joint data, use the following command:

```bash
python3 main.py --train-data-path data/ntu/<dataset folder> --test-data-path data/ntu/<dataset folder>
```

Here <dataset folder> refers to the folder containing the tfrecord files generated in step 5 of the pre-processing steps.

**Note:** At the moment, only `nturgbd-cross-subject` is supported.

## Citation

Please cite the following paper if you use this repository in your reseach    

    @inproceedings{yan2018spatial,
          title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
          author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
          booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
          year={2018}
    }
