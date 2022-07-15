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

    def forward(self, x365, x351):
        x366=operator.add(x365, x351)
        return x366

m = M().eval()
x365 = torch.randn(torch.Size([1, 160, 14, 14]))
x351 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x365, x351)
end = time.time()
print(end-start)
