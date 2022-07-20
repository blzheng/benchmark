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

    def forward(self, x808, x804):
        x809=x808.sigmoid()
        x810=operator.mul(x804, x809)
        return x810

m = M().eval()
x808 = torch.randn(torch.Size([1, 3840, 1, 1]))
x804 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x808, x804)
end = time.time()
print(end-start)
