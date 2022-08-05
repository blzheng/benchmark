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
        self.layernorm5 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x55):
        x56=self.layernorm5(x55)
        return x56

m = M().eval()
x55 = torch.randn(torch.Size([1, 28, 28, 384]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
