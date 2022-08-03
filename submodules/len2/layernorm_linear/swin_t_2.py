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
        self.linear4 = Linear(in_features=384, out_features=192, bias=False)

    def forward(self, x55):
        x56=self.layernorm5(x55)
        x57=self.linear4(x56)
        return x57

m = M().eval()
x55 = torch.randn(torch.Size([1, 28, 28, 384]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
