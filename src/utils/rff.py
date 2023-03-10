import torch
from torch import nn
import math


def sample_xi(nin, nout, sigma=1):
    """
    Generates a tensor of size (nout, nin) with values drawn from a normal distribution 
    with mean 0 and standard deviation sigma.

    Args:
        nin (int): Number of input features.
        nout (int): Number of output features.
        sigma (float, optional): Standard deviation of the normal distribution. Defaults to 1.

    Returns:
        torch.Tensor: Randomly generated tensor of size (nout, nin).
    """
    return torch.randn((nin, nout)) * sigma

@torch.jit.script
def RandomFourierFeatureMap(xi: torch.Tensor, x: torch.Tensor):
    """
    Computes a random Fourier feature map using cosine functions.

    Args:
        xi (torch.Tensor): Random matrix of size (nout, nin), where nin is the number of input features 
                            and nout is the number of output features.
        x (torch.Tensor): Input tensor of size (batch_size, nin), where batch_size is the number of 
                          input examples and nin is the number of input features.

    Returns:
        torch.Tensor: Tensor of size (batch_size, nout) obtained by applying the random Fourier feature 
                      map to the input tensor x.
    """
    assert xi.shape[0] == x.shape[1], f"The second dimension of xi {xi.shape} must match the second dimension of x {x.shape}."
    z = torch.matmul(x, xi) # Matrix multiplication between x and the transpose of xi
    return torch.cos(z) # Apply cosine function element-wise to the resulting matrix

class RandomFourierFeatureLayer(nn.Module):

    """
    PyTorch module for generating random Fourier features.

    Attributes:
        xi (torch.Tensor): Random matrix of size (nout, nin), where nin is the number of input features 
                           and nout is the number of output features.
    """

    def __init__(self, nin, nout, sigma=1) -> None:
        """
        Initializes the RandomFourierFeatureLayer module.

        Args:
            nin (int): Number of input features.
            nout (int): Number of output features.
            sigma (float, optional): Standard deviation of the normal distribution. Defaults to 1.
        """
        super().__init__()
        self.InvSqrtD = 1 / math.sqrt(nout)
        self.xi = nn.Parameter(sample_xi(nin, nout, sigma), requires_grad=False)

    def forward(self, x):
        """
        Computes the output of the RandomFourierFeatureLayer module.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, nin), where batch_size is the number of 
                              input examples and nin is the number of input features.

        Returns:
            torch.Tensor: Tensor of size (batch_size, nout) obtained by applying the random Fourier feature 
                          map to the input tensor x using the precomputed xi buffer.
        """
        return self.InvSqrtD * RandomFourierFeatureMap(self.xi, x)