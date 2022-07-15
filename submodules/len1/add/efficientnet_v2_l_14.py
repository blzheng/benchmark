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

    def forward(self, x109, x103):
        x110=operator.add(x109, x103)
        return x110

m = M().eval()
x109 = torch.randn(torch.Size([1, 96, 28, 28]))
x103 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x109, x103)
end = time.time()
print(end-start)
