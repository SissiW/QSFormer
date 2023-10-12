# 《Few-Shot Learning Meets Transformer: Unified Query-Support Transformers for Few-Shot Classification》
TCSVT 2023

[paper](https://ieeexplore.ieee.org/abstract/document/10144072) &nbsp;&nbsp;

## Abstract 
The goal of Few-shot classification (FSL) is to identify unseen classes with very limited samples has attracted more and more attention. Usually, it is formulated as a metric learning problem. The core issue of few-shot classification is how to learn (1) 
consistent representations for images in both
support and query sets and (2) effective metric learning for
images between support and query sets. In this paper, we show
that the two challenges can be well modeled simultaneously via a
unified Query-Support TransFormer (QSFormer) model. To be
specific, the proposed QSFormer involves global query-support
sample Transformer (sampleFormer) branch and local patch
Transformer (patchFormer) learning branch. sampleFormer aims to capture the dependence of samples in support and query sets
for image representation. It adopts the Encoder, QS-Decoder and
Cross-Attention to respectively model the Support, Query (image)
representation and Metric learning for few-shot classification
task. Also, as a complementary to global learning branch, we
adopt a local patch Transformer to extract structural representation for each image sample by capturing the long-range
dependence of local image patches. In addition, we introduce a
novel Cross-scale Interactive Feature Extractor (CIFE) to extract
and fuse different scale CNN features as an effective backbone
module for the proposed few-shot learning method. We integrate
these into a unified framework and train it in an end-to-end
way. A large number of experiments are conducted on four
popular datasets to validate the superiority and effectiveness of
the proposed QSFormer.

## Architecture
![overview](https://github.com/SissiW/QSFormer/blob/main/overview.png)

## Results on MiniImageNet and TieredImageNet
More experimental results can be found in the paper.
![results](https://github.com/SissiW/QSFormer/blob/main/mini_tiered_result.png?raw=true)

## Datasets
We perform the abundant experiments on four
publicly popular datasets for few-shot classification task,
such as miniImageNet, tieredImageNet, Fewshot-CIFAR100 and Caltech-UCSD Birds-200-2011.
These datasets can be downloaded to click Baidu Drive ([miniImageNet](https://pan.baidu.com/s/1yTn78HgbkrRh_3EClax5FA) (password: rqcs), [tieredImageNet](https://pan.baidu.com/s/1Z9ZsYkwAY11Z_Glzu4tChQ) (password: k5z6), [FC100](https://pan.baidu.com/s/1atEdnikzs8zfKXuO4xr1rQ) (password: 3cib), [CUB](https://pan.baidu.com/s/1defYYyFQL5ZV1Dzug5paHQ) (password: qkpc))

## Installation
python3.7+, pytorch>=1.7, qpth, CVXPY, OpenCV-python, tensorboard

## Download Pre-trained Models
[Baidu Drive](https://pan.baidu.com/s/1UWnpjNaaCTSUB2sOtJqZng)
提取码：yd8w

## Config
```
sh train_meta_QSFormer.sh
```


## Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@article{wang2023few,
  title={Few-Shot Learning Meets Transformer: Unified Query-Support Transformers for Few-Shot Classification},
  author={Wang, Xixi and Wang, Xiao and Jiang, Bo and Luo, Bin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
This project is built upon [DeepEMD](https://github.com/icoz69/DeepEMD). We also reference some code from [DETR](https://github.com/facebookresearch/detr). Thanks to the contributors of these great codebases.
