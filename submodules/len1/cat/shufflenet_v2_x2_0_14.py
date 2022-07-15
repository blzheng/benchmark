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

    def forward(self, x320, x329):
        x330=torch.cat((x320, x329),dim=1)
        return x330

m = M().eval()
x320 = torch.randn(torch.Size([1, 488, 7, 7]))
x329 = torch.randn(torch.Size([1, 488, 7, 7]))
start = time.time()
output = m(x320, x329)
end = time.time()
print(end-start)
