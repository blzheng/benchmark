import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.embedding2 = Embedding(512, 128)
        self.layernorm0 = LayerNorm((128,), eps=1e-12, elementwise_affine=True)
        self._tensor_constant10 = torch.rand(torch.Size([1, 384])).to(torch.int64)

    def forward(self, x23):
        x25=self.embedding2(self._tensor_constant10)
        x26=operator.add(x23, x25)
        x27=self.layernorm0(x26)
        return x27

m = M().eval()
x23 = torch.randn(torch.Size([1, 384, 128]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
