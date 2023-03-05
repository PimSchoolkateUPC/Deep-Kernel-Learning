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

# Paper Comments

## Deep hybrid neural-kernel networks using random Fourier features
authors: Siamak Mehrkanoon, Johan A.K. Suykens

1. **Learning Problem**: Classification (Binary and Multiclass)
2. **Kernel method**: RBF kernel:
$$K(x-y) = \int_{\mathbb{R}^d} p(\xi)e^{j\xi^T(x-y)}d\xi = \mathbb{E}_{\xi}(z_{\xi}(x)z_{\xi}(y))$$

