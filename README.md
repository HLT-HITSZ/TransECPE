# Introduction
This repository was used in our paper:  
  
**“Transition-based Directed Graph Construction for Emotion-Cause Pair Extraction”**  
Chuang Fan, Chaofa Yuan, Jiachen Du, Lin Gui, Min Yang, Ruifeng Xu. ACL 2020
  
Please cite our paper if you use this code.  
# Prerequisites
Python 3.6  
[Pytorch](https://pytorch.org/) 1.1.0  
[CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) 10.0  
BERT - Our bert model is adapted from this implementation: https://github.com/huggingface/pytorch-pretrained-BERT  
# Descriptions
**Data** - A dir where contains resources used in this code.  
* ```bert-base-chinese```: Put the download Pytorch bert model here. 
* ```DataSplits```: A dir where contains 20 different training/validation/test splits in a ratio of 8:1:1. Each sub-dir contains four file: saved_results.txt, train.pkl, valid.pkl and test.pkl.  
  * ```saved_results.txt```: The results of test set for emotion extraction, cause extraction and emotion-cause pair extraction. We adopt early stopping strategy, and the highest F-measure model on the validation set is used to evaluate the test set.  
  * ```train.pkl```: A list where contains two items. train\[0\] is a list of document and train\[1\] is a list the correspondding emotion-cause pairs. For example, train\[0\]\[0\]="*Last week, I lost my phone where shopping, I feel sad now*", then train\[1\]\[0\]=\[(2, 1)\].  
  * ```valid.pkl```: Similar to train.pkl.  
  * ```test.pkl```: Similar to train.pkl.  
* ```doc2pair.pkl```: A dict where the key is the content of a document, and the value is the correspondding emotion-cause pairs.  

**Utils** - A dir where contains several python scripts used in this code.  
* ```Evaluation.py```: Used to evaluate the performance of the proposed model.  
* ```Metrics.py```: Metrics for emotion extraction, cause extraction and emotion-cause pair extractions.  
* ```PrepareData.py```: The scipt for preparing data.  
* ```Transform.py```: Transforming documents to a sequence of defined actions and parser states from left-to-right based on the emotion-cause pairs.  

```Config.py``` - The script holds all the model configuration.  
```TransModule.py``` - The script where contains the proposed transition-based model.  
```Run.py``` - The main script to train and evaluate the proposed transition-based model on different splits.  
# Usage
python3 Run.py
