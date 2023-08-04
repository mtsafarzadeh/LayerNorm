import torch.nn as nn
import enum

class LayerNormType(enum.Enum):
    L2 = enum.auto()
    L1 = enum.auto()
    L1_MEAN_ABS = enum.auto()
    
class LayerNorm(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs. For EXPERIMENTAL purposes only - does not support
    PNNF conversion and works only in FP mode

    Args:
        name (str): name of the modul
        input_a (str): name of the input module or external input
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5

        >>> m = LayerNorm(name='layernorm', input_a='embedding', embedding_dim) # calculate LN over last dimension
        >>> output = m(input_a)
    """

    def __init__(
        self,
        name: str,
        input_a: str,
        shape: Union[int, Tuple[int], List[int], torch.Size],
        eps: float = 1e-5,
        layer_norm_type: LayerNormType = LayerNormType.L2,
        **kwargs,
    ):
        super().__init__(name=name, input_a=input_a, ignore_in_pnnf=True, **kwargs)
        self.core = nn.LayerNorm(shape, eps, **kwargs)
        self.layer_norm_type = layer_norm_type
        if self.layer_norm_type == LayerNormType.L1:
            self.core = L1LayerNorm(shape, eps, **kwargs)
        elif self.layer_norm_type == LayerNormType.L1_MEAN_ABS:
            self.core = MeanAbsLayerNorm(shape, eps, **kwargs)
        else:
            self.core = nn.LayerNorm(shape, eps, **kwargs)
            
class L1LayerNormBase(nn.Module):
    def __init__(
        self,
        normalized_shape: tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        """Base class for implementing L1 version of Layernorm as opposed to default Pytorch layernorm which is L2.

        The usage is the same as Pytorhc layernorm: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normalized_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param eps: Epsilon for calculating variance.
        :param elementwise_affine: Add weight and bias parameters if it is True.
        The forward method should be overwritten by its subclasses.
        """
        super().__init__()
        self.epsilon = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @abstractmethod
    def forward(self, x):
        pass


class L1LayerNorm(L1LayerNormBase):
    def __init__(self, normalized_shape: tuple, eps=1e-5, elementwise_affine=True):
        """L1 Layer normalization for the incoming activations."""
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        dev = x - mean
        denom = (torch.abs(dev)).mean(dim=-1, keepdim=True) + self.epsilon
        y = dev / denom
        if self.weight is not None:
            y *= self.weight
        if self.bias is not None:
            y += self.bias
        return y


class MeanAbsLayerNorm(L1LayerNormBase):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """Mean Abs version of L1 Layer normalization for the incoming activations."""
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        denom = (torch.abs(x)).mean(dim=-1, keepdim=True) + self.epsilon
        y = x / denom
        if self.weight is not None:
            y *= self.weight
        if self.bias is not None:
            y += self.bias
        return y
