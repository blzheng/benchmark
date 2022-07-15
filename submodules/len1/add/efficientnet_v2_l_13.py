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

    def forward(self, x102, x96):
        x103=operator.add(x102, x96)
        return x103

m = M().eval()
x102 = torch.randn(torch.Size([1, 96, 28, 28]))
x96 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x102, x96)
end = time.time()
print(end-start)
