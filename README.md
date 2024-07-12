## Introduction

We train Top K sparsity based Sparse Autoencoder on ResNet50 model, using the weights ```ResNet50_Weights.IMAGENET1K_V2```.

To simplify the problem, we train only the activations of ```maxpool``` layer, for 0th channel ```[0,0,:32,:32]``` with an expansion factor of ```1``` and ```k=1,2,4,8,16,32```.

![ResNet50 Architecture](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*VM94wVftxP7wkiKo4BjfLA.png)

The dataset is taken from ```ILSVRC/imagenet-1k```.



## Table of shapes


| Layer Name | Output Shape | Channel size | 
|------------|---------------|-------------|
| maxpool | [1, 64, 56, 56] | 3136 | 
| layer1 | [1, 256, 56, 56] | 3136 |
| layer2 | [1, 512, 28, 28] | 784 |
| layer3 | [1, 1024, 14, 14] | 196 | 
| layer4 | [1, 2048, 7, 7] | 49 |

