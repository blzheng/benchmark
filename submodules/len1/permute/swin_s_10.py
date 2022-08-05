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

    def forward(self, x231):
        x232=x231.permute(2, 0, 1)
        return x232

m = M().eval()
x231 = torch.randn(torch.Size([49, 49, 12]))
start = time.time()
output = m(x231)
end = time.time()
print(end-start)
