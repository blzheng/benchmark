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

    def forward(self, x271, x256):
        x272=operator.add(x271, x256)
        return x272

m = M().eval()
x271 = torch.randn(torch.Size([1, 176, 14, 14]))
x256 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x271, x256)
end = time.time()
print(end-start)
