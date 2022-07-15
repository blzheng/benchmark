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

    def forward(self, x685, x670):
        x686=operator.add(x685, x670)
        return x686

m = M().eval()
x685 = torch.randn(torch.Size([1, 304, 7, 7]))
x670 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x685, x670)
end = time.time()
print(end-start)
