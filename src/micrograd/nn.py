import random
import math
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, activation_function='relu', eps = 0):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.act_f = activation_function
        self.eps = eps

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.act_f == "relu":
            return act.relu()
        elif self.act_f == "rff":
            return act.RFF(self.eps)
        elif self.act_f == "linear":
            return act
        else:
            raise ValueError(f'"{self.act_f}" is not an existing activation function')

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.act_f == 'relu' else 'RFF' if self.act_f == 'rff' else 'Linear'}Neuron({len(self.w)})"


class LinearNeuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        return sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"LinearNeuron({len(self.parameters())})"


class CosineNeuron(Module):

    """Implementation of Random Fourier Feature Cosine function based on Mehrkanoon"""

    def __init__(self, nin, nout, sigma=1, static=True):
        self.w = [Value(random.gauss(0, sigma)) for _ in range(nin)]
        self.sigma = sigma
        self.R = nout
        self.static = static

    def __call__(self, x):
        if self.static:
            return 1 / math.sqrt(self.R) * sum(wi*xi for wi, xi in zip(self.w, x)).cos()
        return 1 / math.sqrt(self.R)  * sum(random.gauss(0, self.sigma)*xi for xi in x).cos()

    def __repr__(self):
        return f"CosineNeuron({len(self.w)})"


# class Activation():

#     def __init__(self):
#         self.name = "Activation function"
#         self.hypers = {}

#     def __repr__(self):
#         return f"{self.name} with {len(self.hypers)} hyperparameters"


# class ReLU(Activation):

#     def __init__(self):
#         self.name = 'ReLU Activation function'

#     def __call__(self, x: Value):
#         out = Value(0 if x.data < 0 else x.data, (x,), 'ReLU')

#         def _backward():
#             self.grad += (out.data > 0) * out.grad
#         out._backward = _backward

#         return out

# class RFF():

#     def __init__(self, sigma=1, D=1000, width=100):
#         self.name = 'Random Fourier Feature Layer'
#         self.hypers = {f"eps_{i+1}": random.randn(0, sigmas) for i in range(width)}
#         self.hypers['sqrtD'] = math.sqrt(D)

#     def __call__(self, x: Value):
#         out = (1 / hypers['sqrtD']) * sum(Value(math.cos(hypers[f"eps_{i+1}"] * x)) for i in range(width))

#         def _backward():
#             self.grad += (1 / hypers['sqrtD']) * sum(-hypers[f"eps_{i+1}"] * math.sin(-hypers[f"eps_{i+1}"] * self.data)) * out.grad


class RandomFourierFeatureLayer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [CosineNeuron(nin, nout, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"layer of [{', '.join(str(n) for n in self.neurons)}]"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class LinearLayer(Layer):
    
    def __init__(self, nin, nout):
        self.neurons = [LinearNeuron(nin) for _ in range(nout)]


class MLP(Module):

    def __init__(self, nin: int, nouts: list, acts=None, act_out='linear'):
        acts = acts if isinstance(acts, type(list)) else ['relu' for _ in range(len(nouts)-1)] + [act_out]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation_function=acts[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class Mehrkanoon_MLP(Module):

    def __init__(self, nin: int, nouts=[2], version=2, act_out='linear', sigmas=None):
        self.version = version
        self.nin = nin
        self.nouts = nouts

        if len(self.nouts) > 1:
            pass
        else:
            self._build_network_version()

    def _build_network_version(self):
        if self.version == 1:
            self.layers = [LinearLayer(self.nin, 100), 
                           RandomFourierFeatureLayer(100, 25, sigma=0.7),
                           LinearLayer(25, self.nouts[0])]
        elif self.version == 2:
            self.layers = [LinearLayer(self.nin, 2), 
                           RandomFourierFeatureLayer(2, 2, sigma=0.7),
                           LinearLayer(2, 2),
                           RandomFourierFeatureLayer(2, 2, sigma=0.8),
                           LinearLayer(1, self.nouts[0])]
        else:
            raise ValueError(f"'version' should be either set to 1 or 2, {version} was given")

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"