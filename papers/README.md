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

## Deep hybrid neural-kernel networks using random Fourier features
authors: Siamak Mehrkanoon, Johan A.K. Suykens

1. **Learning Problem**: Classification (Binary and Multiclass)
2. **Kernel method**: RBF kernel:

```math
K(x-y) = \int_{\mathbb{R}^d} p(\xi)e^{j\xi^T(x-y)}d\xi = \mathbb{E}_{\xi} (z_{\xi}(x)z_{\xi}(y))
```
   Feature map (implemented):

```math
\hat{\varphi}(x) = \frac{1}{\sqrt{D}}\left[z_{\xi_1}(x),...,z_{\xi_D}(x)\right]^T
```
```math
z_{\xi}(x) = cos(\xi^T x)
   ```

   Feature map (Alternative; using kernel trick):
```math
\hat{\varphi}(x) = \frac{1}{\sqrt{\lambda_i}}\sum^m_{k=1}u_{ki}K(x_k, x), \ i = 1,..., m
```
   where $\lambda_i$ and $u_i$ are eigenvalues and eigenvectors of kernel matrix $\Omega_{m\times m}$. Using this as an implementation would require to learn the kernel matrix $\Omega_{m \times m}$ which is **NOT** done in the paper.

3. **Describe how the model is trained**: Stochastic Gradient descent and back propagation.

   Loss function (Cross entropy - For multiclass and Binary):
   ```math
   L(x_i, y_i) = - \log\left(\frac{\exp{\left(s_i^{y_i}\right)}}{\sum^Q_{j=1}\exp{\left(s_i^{y_i}\right)}}\right)
   ```

   Regularization term: (L2 regularization)
   ```math
   \frac{\gamma}{2}\sum^2_{j=1}\mathrm{Tr}\left(W_jW_j^T\right)
   ```

   Cost / Emperical risk / Optimization problem:
   ```math
   \underset{W_1, W_2, b_1, b_2}{\min} J(W_1, W_2, b_1, b_2) = \frac{\gamma}{2}\sum^2_{j=1}\mathrm{Tr}\left(W_jW_j^T\right) + \frac{1}{n}\sum^n_{i=1}L(x_i, y_i)
   ```

4. **Paper has been implemented**

------------------------------------



