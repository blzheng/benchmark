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

    def forward(self, x715):
        x716=x715.mean((2, 3),keepdim=True)
        return x716

m = M().eval()
x715 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x715)
end = time.time()
print(end-start)
