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

    def forward(self, x21, x26):
        x27=operator.mul(x21, x26)
        return x27

m = M().eval()
x21 = torch.randn(torch.Size([1, 96, 56, 56]))
x26 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x21, x26)
end = time.time()
print(end-start)
