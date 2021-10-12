# PyTorch Model Compare (WIP)

A tiny package to compare two neural networks in PyTorch. There are many ways to compare two neural networks, but one robust and scalable way is using the **Centered Kernel Alignment** (CKA) metric, where the features of the networks are compared.

### Centered Kernel Alignment
Centered Kernel Alignment (CKA) is a representation similarity metric that is widely used for understanding the representations learned by neural networks. Specifically, CKA takes two feature maps / representations X and Y as input and computes their normalized similarity (in terms of the Hilbert-Schmidt Independence Criterion (HSIC)) as


However, the above formula is not scalable against deep architectures and large datasets. Therefore, a minibatch version can be constructed that uses an unbiased estimator of the HSIC as

![alt text](assets/cka_mb.png "CKA minibatch version")

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

### Comparing two ResNets

### Comparing two similar architectures

### Comparing a ResNet with Vision Transformer (ViT)

### Comparing Dataset Drift 






