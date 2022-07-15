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

    def forward(self, x162, x148):
        x163=operator.add(x162, x148)
        return x163

m = M().eval()
x162 = torch.randn(torch.Size([1, 160, 7, 7]))
x148 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x162, x148)
end = time.time()
print(end-start)
