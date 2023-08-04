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
