# PT_propagation_then_training

This is the official pytorch implementation of Propagation then Training (PT), a graph algorithm. PT is proposed in the paper: 

[On the Equivalence of Decoupled Graph Convolution Network and Label Propagation](https://arxiv.org/abs/2010.12408) 

by Hande Dong, Jiawei Chen, Fuli Feng, Xiangnan He, Shuxian Bi, Zhaolin Ding, Peng Cui

Published at WWW 2021. 

## Introduction

In this work, we propose a two-step label propagation algorithm: (1) propagating the label of training set, (2) training a base predictor supervised by the soft label obtained in the previous step. Through gradient analysis, we proved that it is equivalent to decoupled GCN. 

## Environment Requirement

The code runs well under python 3.8.5. The required packages are as follows: 

- pytorch == 1.6.0
- numpy == 1.19.1
- scipy == 1.5.2

## Datasets

Following [APPNP](https://github.com/klicperajo/ppnp), we use four datasets:  CITESEER,  CORA_ML,  PUBMED and MS_ ACADEMIC. 

CITESEER, CORA_ML, and PUBMED are citation networks, where each node represents a paper and an edge indicates a citation relationship. MS_ACADEMIC is a co-authorship network, where nodes and edges represent authors and co-author relationship, respectively. 

## Run the Code

To run **Propagation then Training Adaptively** (PTA), execute the following command: 

```shell
python train_PT.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.0 --epochs 2000 --hidden 64 --lr 0.1 --weight_decay 0.005 --loss_decay 0.05 --fast_mode False --mode 2 --epsilon 100 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

To run **Propagation then Training Statically** (PTS, a variant of PT), execute the following command: 

```shell
python train_PT.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.0 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --loss_decay 0.05 --fast_mode False --mode 0 --epsilon 100 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

To run **Propagation then Training Dynamically** (PTD, a variant of PT), execute the following command: 

```shell
python train_PT.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.0 --epochs 2000 --hidden 64 --lr 0.1 --weight_decay 0.005 --loss_decay 10 --fast_mode False --mode 1 --epsilon 100 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

To run **APPNP** (our main baseline), execute the following command: 

```shell
python train_APPNP.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

- Above commands are for Cora_ML dataset. For Citeseer or Pubmed, replace '--dataset cora_ml' with '--dataset citeseer' or '--dataset pubmed'. For MS_Academic, replace '--dataset cora_ml' with '--dataset ms_academic' and replace '--alpha 0.1' with '--alpha 0.2'. 
- PTS, PTD, and PTA are three variants of PT. The argument '--mode' is for the three variants by setting 0/1/2. PTA performs best among them. 
- The argument '--str_noise_rate' is for specifying structure noise rate in graph. To keep the original graph, set '--str_noise_rate' to 2.0. 
- The argument '--lbl_noise_num' is for specifying label noise numbers of training set. To keep the original labels, set '--lbl_noise_num' to 0. 

## Contact

Please contact donghd@mail.ustc.edu.cn if you have any questions about the code and paper. 

## Reference
This code refers to: 

https://github.com/klicperajo/ppnp

https://github.com/tkipf/gcn/blob/master/gcn/utils.py