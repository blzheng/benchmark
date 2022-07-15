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

    def forward(self, x284, x279):
        x285=operator.mul(x284, x279)
        return x285

m = M().eval()
x284 = torch.randn(torch.Size([1, 1248, 1, 1]))
x279 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x284, x279)
end = time.time()
print(end-start)
