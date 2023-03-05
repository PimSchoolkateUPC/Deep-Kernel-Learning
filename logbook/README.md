# LogBook of thesis process
This logbook is intented for evaluation of the TFM and progress tracking purposes.
It was initiliazed on: 03/05/2023.

## 02/01/2023 - 03/05/2023
In no particular order

 - Learnt PyTorch.
   - Basic building of sequential models
   - Customizable layers
   - Customizable Loss functions
 
 - Implemented basic dataset loader for datasets used in "Deep hybrid neural-kernel networks using random Fourier features" compatible with PyTorch. This will serve a basic evaluation method for upcoming model designs. Adaptable for including more datasets. See [src/utils/Dataloader.py](https://github.com/PimSchoolkateUPC/Deep-Kernel-Learning/blob/main/src/utils/dataloader.py) (folder structure might be renamed later).

 - Gathered related works on Google Scholar

 - Read papers:
   - Deep Layer-wise Networks Have Closed-Form Weights
   - Deep hybrid neural-kernel networks using random Fourier features
   - Deep Kernel Learning
   
 - Build implementation for the papers:
   - Deep Layer-wise Networks Have Closed-Form Weights [WORK IN PROCESS][NOT WORKING]
     - Worked on understanding original implementation
     - compared original implementation with paper; do not allign; worked on understanding discrepencies
     - 
   - Deep hybrid neural-kernel networks using random Fourier features [WORK IN PROCESS][WORKING]
     - Implemenation by extending Micrograd for the specific RFF case (https://github.com/karpathy/micrograd). This was done to properly understand the inner workings of the specific architecture, as well as a refresher on Deep Neural Nets after being spoiled by TensorFlow. Extremely slow as it uses for loops instead of matrix multiplication.
     - Implemenation in PyTorch. Fast reliable implementation.
   
 - Started exploring and implementing Kernel Methods for future models
   - RBF kernel
   - Linear Kernel
   - Sigmoid Kernel
   - Relu Kernel [NO DOTPRODUCT]
   - LeakyRelu Kernel [NO DOTPRODUCT]

## 03/05/2023
 - Added read and unread papers to repo + restructured repo
 - Devised reading strategy for unread papers
 - Applied Strategy to already read papers (see above)
 - Implemented Spectral Mixture Kernel and feature map as defined by Deep Kernel Learning.
