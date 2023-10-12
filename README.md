# 《Few-Shot Learning Meets Transformer: Unified Query-Support Transformers for Few-Shot Classification》TCSVT 2023

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
![results](https://github.com/SissiW/QSFormer/blob/main/mini_tiered_result.png?raw=true)


