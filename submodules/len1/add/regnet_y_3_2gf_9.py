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

    def forward(self, x153, x167):
        x168=operator.add(x153, x167)
        return x168

m = M().eval()
x153 = torch.randn(torch.Size([1, 576, 14, 14]))
x167 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x153, x167)
end = time.time()
print(end-start)