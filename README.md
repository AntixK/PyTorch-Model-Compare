# PyTorch Model Compare (WIP)

A tiny package to compare two neural networks in PyTorch. There are many ways to compare two neural networks, but one robust and scalable way is using the **Centered Kernel Alignment** (CKA) metric, where the features of the networks are compared.

### Centered Kernel Alignment
Centered Kernel Alignment (CKA) is a representation similarity metric that is widely used for understanding the representations learned by neural networks. Specifically, CKA takes two feature maps / representations X and Y as input and computes their normalized similarity (in terms of the Hilbert-Schmidt Independence Criterion (HSIC)) as

<img src="assets/cka.png" alt="CKA original version" width="75%">

However, the above formula is not scalable against deep architectures and large datasets. Therefore, a minibatch version can be constructed that uses an unbiased estimator of the HSIC as

![alt text](assets/cka_mb.png "CKA minibatch version")

![alt text](assets/cka_hsic.png "CKA HSIC calculation")

The above results follow directly from the 2021 ICLR paper by [Nguyen T., Raghu M, Kornblith S](https://arxiv.org/abs/2010.15327).

## Getting Started

### Installation
```
pip install torch_cka
```
### Usage
```python
from torch_cka import CKA
model1 = resnet18(pretrained=True)
model2 = resnet34(pretrained=True)

dataloader = DataLoader(your_dataset, 
                        batch_size=batch_size, 
                        shuffle=False)

cka = CKA(model1, model2,
          device='cuda')

cka.compare(dataloader)
```

## Examples
`torch_cka` can be used with any pytorch model (subclass of `nn.Module`) and can be used with pretrained models available from popular sources like torchHub, timm, huggingface etc. Some examples of where this package can come in handy are illustrated below.

### Comparing two ResNets

<img src="assets/resnet_compare.png" alt="Comparing ResNet18 and ResNet34" width="75%">

### Comparing two similar architectures

### Comparing a ResNet with Vision Transformer (ViT)

### Comparing Dataset Drift 






