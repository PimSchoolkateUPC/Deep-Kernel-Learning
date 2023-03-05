# Reading Strategy:

1. define learning problem
2. Mathematically describe what kernel method(s) is/are used.
   - Dot product as well as feature map
   - Kernel composition
3. Describe how the model is trained
   - Optimization method (Layer-wise vs Backprop)
   - Loss function: What is being measured?
   - Regularization methods
4. Formulate if it is easy to implement based on current implemenations

-------------------------------------

# Paper Comments

## Deep Layer-wise Networks Have Closed-Form Weights
authors: Chieh Wu, Aria Masoomi, Arthur Gretton, Jennifer Dy

Implements feature map of a kernel as an activation function for all of the layers of the network. In the implemenation kernels are used to compute the loss function. Network is approximately infinite (using RFF)

1. **Learning Problem**: Classification (Binary and Multiclass)
2. **Kernel method**: RBF kernel
   Feature map: Radial Basis Function using Random Fourier Features. With Phase shift

3. **Describe how the model is trained**: Layer wise - closed form weight optimization
   **Loss function** Hilbert Schmidth Independence Criterion
   **Regularization term**: Claim: HSIC regularizes naturally

4. **Paper has been implemented**

Comments:
- Contains methods for expressing shattering of data before and after activation function
- Contains proof for closed form weight optimization ONLY for RBF kernel
- 

------------------------------------

## Deep hybrid neural-kernel networks using random Fourier features
authors: Siamak Mehrkanoon, Johan A.K. Suykens

Implements feature map of a kernel as activation function for some of the layers of a network. In the implementation kernels are not being used. Network is approximately infinite (using RFF).

1. **Learning Problem**: Classification (Binary and Multiclass)
2. **Kernel method**: RBF kernel
   Feature map: Radial Basis Function using Random Fourier Features. No phase shift!

3. **Describe how the model is trained**: Stochastic Gradient descent and back propagation for each layer. For deeper networks transfer learning is used.
   **Loss function** (Cross entropy - For multiclass and Binary).
   **Regularization term**: (L2 regularization)

4. **Paper has been implemented**

------------------------------------

## Deep Kernel Learning
authors: Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, Eric P. Xing

Implements a Gaussian Process on top of an already trained DNN. In the implementation kernels are being used, but as an approximation using KISS-GP to reduce computation time. Final layer (GP) is infinite.

1. **Learning Problem**: Classification (Binary and Multiclass) and Regression
2. **Kernel method**: RBF kernel & Spectral Mixture
   Feature map: Radial Basis Function using Random Fourier Features. No phase shift!

3. **Describe how the model is trained**: First pretrain using Stochastic Gradient descent and back propagation for DNN only. Then backpropagation for the kernel method (combined with chain rule) and weight varaibles for the whole pretrained DNN.
   **Loss function** First squared loss. Then Marginal Likelihood of the GP.
   **Regularization term**: It is stated in the related work section that this is not needed in GP.
   
4. Paper has not been implemented. It could potentially be added to the current framework, however, it would require building a framework for training GPs in PyTorch that can sit on top of the current implementation.

Possibly interesting to combine with layer-wise training to create actual infinte networks.

------------------------------------



