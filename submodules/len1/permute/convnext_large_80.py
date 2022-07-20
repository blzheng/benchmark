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

    def forward(self, x421):
        x422=x421.permute(0, 2, 3, 1)
        return x422

m = M().eval()
x421 = torch.randn(torch.Size([1, 1536, 1, 1]))
start = time.time()
output = m(x421)
end = time.time()
print(end-start)
