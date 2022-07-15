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

    def forward(self, x22, x24):
        x25=torch.cat([x22, x24], 1)
        return x25

m = M().eval()
x22 = torch.randn(torch.Size([1, 128, 27, 27]))
x24 = torch.randn(torch.Size([1, 128, 27, 27]))
start = time.time()
output = m(x22, x24)
end = time.time()
print(end-start)
