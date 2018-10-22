
# About

This repository is a fork of [dichotomies/proxy-nca](https://github.com/dichotomies/proxy-nca), which contains a PyTorch implementation of [`No Fuss Distance Metric Learning using Proxies`](https://arxiv.org/pdf/1703.07464.pdf) as introduced by Google Research.

The fork adds training for a second dataset (UPMC-G20 containing 20 food categories with 100 images per category)

The same parameters were used as described in the paper, except for the optimizer. In particular, the size of the embedding and batches equals 64 and 32 respectively. Also, [BN-Inception](http://arxiv.org/abs/1502.03167) is used and trained with random resized crop and horizontal flip and evaluated with resized center crop. 

The [PyTorch BN-Inception model](https://github.com/Cadene/pretrained-models.pytorch) has been ported from PyTorch 0.2 to 0.4. Its weights are stored inside the repository in the directory `net`.

# Training with UPMC-G20

You need Python3 and minimum PyTorch 0.4.1 to run the code via the following command: `python3 train.py`. This will download the UPMC-G20 dataset automatically and perform training for 20 epochs.

| Metric |   Results  |
| ------ | ---------- |
|  R@1   |    96.45   | 
|  R@2   |    97.55   |
|  R@4   |    98.05   |
|  R@8   |    98.25   |
|  NMI   |    91.74   |

# Reproducing Results with CUB 200

If you want to reproduce the CUB200 results in the table below, then the only thing you have to do is to execute: `python3 train.py --dataset CUB200 --root-folder cub200 --number-classes 100`.

In this case, the CUB dataset will be automatically downloaded to the directory `cub200` and verified with the corresponding md5 hash. If you train the model for the first time, then the images file will be extracted automatically in the same folder. After that you can use the argument `--cub-is-extracted` to avoid extracting the dataset over and over again.

Training takes about 10 minutes with one Titan X (Pascal).

| Metric | This Implementation  | [Google's Implementation](https://arxiv.org/pdf/1703.07464.pdf) |
| ------ | -------------------- | ------------- |
|  R@1   |       **49.26**      |     49.21     |
|  R@2   |         60.99        |   **61.90**   |
|  R@4   |       **71.31**      |     67.90     |
|  R@8   |       **80.78**      |     72.40     |
|  NMI   |         58.12        |   **59.53**   |

An example training log file can be found in the log dir, see [`example.log`](https://github.com/dichotomies/proxy-nca/raw/master/log/example.log).
