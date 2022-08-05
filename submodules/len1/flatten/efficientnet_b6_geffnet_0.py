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

    def forward(self, x669):
        x670=x669.flatten(1)
        return x670

m = M().eval()
x669 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x669)
end = time.time()
print(end-start)
