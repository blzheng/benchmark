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

    def forward(self, x296):
        x297=x296.chunk(2,dim=1)
        return x297

m = M().eval()
x296 = torch.randn(torch.Size([1, 976, 7, 7]))
start = time.time()
output = m(x296)
end = time.time()
print(end-start)
