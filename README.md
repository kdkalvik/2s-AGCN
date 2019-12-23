# 2s-AGCN
An **unofficial** Tensorflow implementation of the paper "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition" in CVPR 2019.

**NOTE**: Experiment results are not being updated due to hardware limits.

- Paper: [PDF](https://pdfs.semanticscholar.org/e48f/36aacb72adb74cef077c87d2351121124137.pdf?_ga=2.28526716.1745611754.1577059428-1756285583.1573525636)

## Dependencies

- Python >= 3.5
- scipy >= 1.3.0
- numpy >= 1.16.4
- tensorflow >= 2.0.0

## Directory Structure

Most of the interesting stuff can be found in:
- `model/agcn.py`: model definition of AGCN
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

4. Then, in `data_gen/`, generate the bone dataset:

```bash
python3 gen_bone_data.py
```

5. Generate the tfrecord files for motion and spatial data :

```bash
python3 gen_tfrecord_data.py
```

The generation scripts look for generated data in previous step. By default they look at `./data`; change dir configs if needed.

## Training

To start training the network with the joint data, use the following command:

```bash
python3 main.py --train-data-path data/ntu/<dataset folder> --test-data-path data/ntu/<dataset folder>
```

Here <dataset folder> refers to the folder containing the tfrecord files generated in step 5 of the pre-processing steps.

**Note:** At the moment, only `nturgbd-cross-subject` is supported.

## Citation

Please cite the following paper if you use this repository in your reseach.

    @inproceedings{2sagcn2019cvpr,  
          title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
          author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
          booktitle = {CVPR},  
          year      = {2019},  
    }

    @article{shi_skeleton-based_2019,
        title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
        journal = {arXiv:1912.06971 [cs]},
        author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
        month = dec,
        year = {2019},
	}
