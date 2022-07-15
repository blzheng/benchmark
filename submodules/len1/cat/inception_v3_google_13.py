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

    def forward(self, x301, x304):
        x305=torch.cat([x301, x304], 1)
        return x305

m = M().eval()
x301 = torch.randn(torch.Size([1, 384, 5, 5]))
x304 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x301, x304)
end = time.time()
print(end-start)
