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

    def forward(self, x148, x150):
        x151=operator.add(x148, x150)
        return x151

m = M().eval()
x148 = torch.randn(torch.Size([1, 2048, 7, 7]))
x150 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x148, x150)
end = time.time()
print(end-start)
