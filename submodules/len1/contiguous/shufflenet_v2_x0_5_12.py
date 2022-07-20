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

    def forward(self, x294):
        x295=x294.contiguous()
        return x295

m = M().eval()
x294 = torch.randn(torch.Size([1, 96, 2, 7, 7]))
start = time.time()
output = m(x294)
end = time.time()
print(end-start)
