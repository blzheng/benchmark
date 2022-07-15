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

    def forward(self, x388, x383):
        x389=operator.mul(x388, x383)
        return x389

m = M().eval()
x388 = torch.randn(torch.Size([1, 2904, 1, 1]))
x383 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x388, x383)
end = time.time()
print(end-start)
