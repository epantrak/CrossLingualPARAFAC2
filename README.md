# Cross-lingual transfer learning: A PARAFAC2 approach

This repository contains the code for the paper "Cross-lingual transfer learning: A PARAFAC2 approach" introduced
by Evangelia Pantraki, Ioannis Tsingalis and Constantine Kotropoulos

## Introduction

The proposed framework addresses the problem of cross-lingual transfer learning resorting to Parallel Factor 
Analysis 2 (PARAFAC2). To avoid the need for multilingual parallel corpora, a pairwise setting is adopted where 
a PARAFAC2 model is fitted to documents written in English (source language) and a different target language. 
Firstly, an unsupervised PARAFAC2 model is fitted to parallel unlabelled corpora pairs to learn the latent 
relationship between the source and target language. The fitted model is used to create embeddings for a text 
classification task (document classification or authorship attribution). Subsequently, a logistic regression 
classifier is fitted to the training source language embeddings and tested on the training target language 
embeddings. Following the zero-shot setting, no labels are exploited for the target language documents. 
The proposed framework incorporates a self-learning process by utilizing the predicted labels as pseudo-labels 
to train a new, pseudo-supervised PARAFAC2 model, which aims to extract latent class-specific information while 
fusing language-specific information.

## Usage
### 1. Requirements
The requirements are in the requirements.txt file. 


### 2. Download Datasets
You can download the dataset from here and extract them to the folder Datasets (will be included).


### 3. Train and Test

To train the general PARAFAC2 model, you can run

```angular2
--target_lang french --masked True --tokenizer nltk --R 2500
```

To test the PARAFAC2 model in MLDOC, you can run
```angular2
--target_lang chinese --target_dataset mldoc --tokenizer nltk --masked True --R 2500
```

## Contact
If you have any question, please feel free to email us.

Evangelia Pantraki (epantrak@csd.auth.gr)
