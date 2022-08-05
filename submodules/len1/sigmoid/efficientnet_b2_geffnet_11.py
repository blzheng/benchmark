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

    def forward(self, x170):
        x171=x170.sigmoid()
        return x171

m = M().eval()
x170 = torch.randn(torch.Size([1, 528, 1, 1]))
start = time.time()
output = m(x170)
end = time.time()
print(end-start)
