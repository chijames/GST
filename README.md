# Gapped Straight-Through Estimator

PyTorch implementation of __Gapped Straight-Through Estimator__ (GST) with experiments on MNIST-VAE and ListOps. We compare our proposed GST estimator with several discrete random variable estimators including Straight-Through Gumbel-Softmax (STGS) and Rao-Blackwellized Straight-Through Gumbel-Softmax (rao_gumbel). 

* Gapped Straight-Through Estimator (GST): Training Discrete Deep Generative Models via Gapped Straight-Through Estimator [ICML 2022].
* Straight-Through Gumbel-Softmax (STGS): [Categorical Reparametrization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf)
* Rao-Blackwellized Straight-Through Gumbel-Softmax (rao_gumbel): [Rao-Blackwellizing the Straight-Through Gumbel-Softmax Gradient Estimator](https://arxiv.org/abs/2010.04838)
* MNIST-VAE experiment is based on [YongfeiYan's implementation](https://github.com/YongfeiYan/Gumbel_Softmax_VAE).
* ListOps experiment is based on [Learning to Compose Task-Specific Tree Structures](https://github.com/jihunchoi/unsupervised-treelstm) and its [implementation](https://github.com/jihunchoi/unsupervised-treelstm). The dataset comes from [ListOps: A Diagnostic Dataset for Latent Tree Learning](https://arxiv.org/abs/1804.06028).

## Installation
We recommend using Anaconda with the following commands:
```
conda create -n GST python=3.8
conda activate GST
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## MNIST-VAE Experiment

#### Configurations
`--mode` (default: gumbel) selects the estimators. Possible choices are gumbel, rao_gumbel, gst-1.0, and gst-p.

`--temperature` (default: 1.0) controls the temperature of the softmax function for the soft sample.

`--hard` (default: True) gives hard samples using the straight-through trick; otherwise, soft samples are generated.

Example 1: train STGS at temperature 1.0
```
python gumbel_softmax_vae.py --mode gumbel --temperature 1.0
```

Example 2: train GST-1.0 at temperature 0.5
```
python gumbel_softmax_vae.py --mode gst-1.0 --temperature 0.5
```

## ListOps Experiment

#### Configurations
Example (a): train rao_gumbel at temperature 0.1
```
python -m nlp.train --word-dim 300 --hidden-dim 300 --clf-hidden-dim 300 --clf-num-layers 1 --batch-size 16 --max-epoch 20 --save-dir ./checkpoint_listops --device cuda --pretrained glove.840B.300d --leaf-rnn --dropout 0.5 --lower --mode rao_gumbel --task listops --temperature 0.1
```

Example (b): train GST-1.0 at temperature 0.1
```
python -m nlp.train --word-dim 300 --hidden-dim 300 --clf-hidden-dim 300 --clf-num-layers 1 --batch-size 16 --max-epoch 20 --save-dir ./checkpoint_listops --device cuda --pretrained glove.840B.300d --leaf-rnn --dropout 0.5 --lower --mode gst-1.0 --task listops --temperature 0.1
```
