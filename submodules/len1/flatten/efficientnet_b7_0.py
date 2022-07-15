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

    def forward(self, x861):
        x862=torch.flatten(x861, 1)
        return x862

m = M().eval()
x861 = torch.randn(torch.Size([1, 2560, 1, 1]))
start = time.time()
output = m(x861)
end = time.time()
print(end-start)
