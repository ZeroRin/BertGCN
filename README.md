# BertGCN
This repo contains code for [BertGCN: Transductive Text Classification by Combining GCN and BERT](https://arxiv.org/abs/2105.05727).


## Introduction

In this work, we propose BertGCN, a model that combines large scale pretraining and transductive learning for text classification. BertGCN constructs a  heterogeneous graph over the dataset and represents documents as nodes using BERT representations. By jointly training the BERT and GCN modules within BertGCN, the proposed model is able to leverage the advantages of both worlds: large-scale pretraining which takes the advantage of the massive amount of raw data and transductive learning which jointly learns representations for both training data and unlabeled test data by propagating label influence through graph convolution. Experiments show that BertGCN achieves SOTA performances on a wide range of text classification datasets. 

## Main Results
|**Model** | **20NG** | **R8** | **R52** | **Ohsumed** | **MR** |
| ------------ | ---- | ---- | ---- | ---- | ---- |
| [*TextGCN*](https://arxiv.org/pdf/1809.05679.pdf) | 86.3 | 97.1 | 93.6 | 68.4 | 76.7 |
| [*SGC*](https://arxiv.org/abs/1902.07153) | 88.5 | 97.2 | 94.0 | 68.5 | 75.9 |
| [*BERT*](https://arxiv.org/abs/1810.04805) | 85.3 | 97.8 | 96.4 | 70.5 | 85.7 |
| [*RoBERTa*](https://arxiv.org/abs/1907.11692) | 83.8 | 97.8 | 96.2 | 70.7 | 89.4 |
| *BertGCN* | 89.3 | 98.1 | **96.6** | **72.8** | 86.0 |
| *RoBERTaGCN* | **89.5** | **98.2** | 96.1 | **72.8** | **89.7**|
| *BertGAT* | 87.4 | 97.8 | 96.5 | 71.2 | 86.5 |
| *RoBERTaGAT* | 86.5 | 98.0 | 96.1 |  71.2 | 89.2 |

## Dependencies

Create environment and install required packages for BertGCN using conda:

`conda create --name BertGCN --file requirements.txt -c default -c pytorch -c dglteam -c huggingface`

If the NVIDIA driver version does not support CUDA 10.1 you may edit requirements.txt to use older cudatooklit and the corresponding [dgl](https://www.dgl.ai/pages/start.html) instead.

## Usage

1. Run `python build_graph.py [dataset]` to build the text graph.

2. Run `python finetune_bert.py --dataset [dataset]` 
to finetune the BERT model over target dataset. The model and training logs will be saved to `checkpoint/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.

3. Run `python train_bert_gcn.py --dataset [dataset] --pretrained_bert_ckpt [pretrained_bert_ckpt] -m [m]`
to train the BertGCN. 
`[m]` is the factor balancing BERT and GCN prediction \(lambda in the paper\). 
The model and training logs will be saved to `checkpoint/[bert_init]_[gcn_model]_[dataset]/` by default. 
Run `python train_bert_gcn.py -h` to see the full list of hyperparameters.

Trained BertGCN parameters can be downloaded [here](https://drive.google.com/file/d/1YUl7q34S3pu8KH17yOI68tvcedkrQ39a).

## Acknowledgement

The data preprocess and graph construction are from [TextGCN](https://github.com/yao8839836/text_gcn)

## Citation
To appear in Findings of ACL 2021
```angular2
@article{lin2021bertgcn,
  title={BertGCN: Transductive Text Classification by Combining GCN and BERT},
  author={Lin, Yuxiao and Meng, Yuxian and Sun, Xiaofei and Han, Qinghong and Kuang, Kun and Li, Jiwei and Wu, Fei},
  journal={arXiv preprint arXiv:2105.05727},
  year={2021}
}
```