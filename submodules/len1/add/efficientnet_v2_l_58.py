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

    def forward(self, x846, x831):
        x847=operator.add(x846, x831)
        return x847

m = M().eval()
x846 = torch.randn(torch.Size([1, 384, 7, 7]))
x831 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x846, x831)
end = time.time()
print(end-start)
