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

    def forward(self, x150, x152):
        x153=operator.add(x150, x152)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 2048, 28, 28]))
x152 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x150, x152)
end = time.time()
print(end-start)
