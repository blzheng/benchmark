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

    def forward(self, x25):
        x26=x25.sigmoid()
        return x26

m = M().eval()
x25 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
