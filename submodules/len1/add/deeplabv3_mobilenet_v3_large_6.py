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

    def forward(self, x106, x98):
        x107=operator.add(x106, x98)
        return x107

m = M().eval()
x106 = torch.randn(torch.Size([1, 80, 14, 14]))
x98 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x106, x98)
end = time.time()
print(end-start)
