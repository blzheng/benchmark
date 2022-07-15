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

    def forward(self, x150, x135):
        x151=operator.add(x150, x135)
        return x151

m = M().eval()
x150 = torch.randn(torch.Size([1, 56, 28, 28]))
x135 = torch.randn(torch.Size([1, 56, 28, 28]))
start = time.time()
output = m(x150, x135)
end = time.time()
print(end-start)
