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
        self.layernorm3 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)

    def forward(self, x48):
        x49=self.layernorm3(x48)
        return x49

m = M().eval()
x48 = torch.randn(torch.Size([1, 28, 28, 192]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
