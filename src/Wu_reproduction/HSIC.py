import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

def HSIC(X, Y, sigma, Kx_type='linear', Ky_type='linear', plot=False):

    n = X.shape[0]

    if Kx_type == 'linear': Kx = X.dot(X.T)
    if Ky_type == 'linear': Ky = Y.dot(Y.T)

    if Kx_type == 'Gaussian':
        gamma = 1.0 / (2 * sigma * sigma)
        Kx = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)

    if plot:
        plt.imshow(Kx, cmap='Blues_r', interpolation='nearest')
        plt.colorbar()
        plt.show()
    
    HKx = Kx - np.mean(Kx, axis=0) # equivalent to		HKᵪ = H.dot(Kᵪ)
    HKy = Ky - np.mean(Ky, axis=0) # equivalent to		HKᵧ = H.dot(Kᵧ)

    Hxy = np.sum(HKx.T*HKy)

    Hx = np.linalg.norm(HKx)
    Hy = np.linalg.norm(HKy)

    hsic = Hxy / (Hx * Hy)

    return [hsic, Kx]

def HSIC2(x, y, sigma, KernelX, KernelY):
    Kx = KernelX(x)
    Ky = KernelY(y)

    HKx = Kx - np.mean(Kx, axis=0) # equivalent to		HKᵪ = H.dot(Kᵪ)
    HKy = Ky - np.mean(Ky, axis=0) # equivalent to		HKᵧ = H.dot(Kᵧ)

    Hxy = np.sum(HKx.T*HKy)

    Hx = np.linalg.norm(HKx)
    Hy = np.linalg.norm(HKy)

    print(f"Hxy: {Hxy}")
    print(f"HKy: {HKy}")
    print(f"Hx: {Hx}")
    print(f"Hy: {Hy}")

    hsic = Hxy / (Hx * Hy)

    return hsic