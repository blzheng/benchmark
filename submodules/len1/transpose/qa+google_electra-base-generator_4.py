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

    def forward(self, x204):
        x216=x204.transpose(-1, -2)
        return x216

m = M().eval()
x204 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)
