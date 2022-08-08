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

    def forward(self, x103, x133):
        x134=operator.add(x103, x133)
        return x134

m = M().eval()
x103 = torch.randn(torch.Size([1, 384, 768]))
x133 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x103, x133)
end = time.time()
print(end-start)
