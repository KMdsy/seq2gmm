# Seq2GMM

## Introduction

The quasi-periodic signals generated by these objects often follow a similar repetitive and periodic pattern, but have variations in period, and come with different lengths as well as jitters and synchronization errors. Given a multitude of such quasi-periodic time series, can we build machine learning models to identify those time series that behave differently from the majority of the observations? In addition, can the models help human experts to understand how the decision was made? We propose a **sequence to Gaussian Mixture Model (seq2GMM)** framework. 
The overarching goal of this framework is to identify unusual and interesting time series within a network time series database. We further develop a surrogate-based optimization algorithm that can efficiently train the seq2GMM model. Seq2GMM exhibits strong empirical performance on a plurality of public benchmark datasets, outperforming state-of-the-art anomaly detection techniques by a significant margin. We also theoretically analyze the convergence property of the proposed training algorithm and provide numerical results to substantiate our theoretical claims.

Seq2GMM is composed of three blocks, namely the temporal segmentation, the temporal compression network, the estimation network with GMM.

![Framework of seq2gmm](img/seq2gmm.png)



## Contributions

- Learning without anomaly training samples
- Robust against synchronization errors
- Visualization and localizaiton
- State of the art

## Requirement

- python>=3.6
- tensorflow1 >= 1.12

## Quick start

```
git clone https://github.com/xxx
cd seq2gmm
```

Train & Test Seq2GMM example

```python
# Train
python main.py

# Test
python test.py
```

The train logs and trained model will be saved under `log` and  `checkpoint` respectively.

We also provide some pre-trained model under `trained_model` .



