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

    def forward(self, x299, x294):
        x300=operator.mul(x299, x294)
        return x300

m = M().eval()
x299 = torch.randn(torch.Size([1, 1056, 1, 1]))
x294 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x299, x294)
end = time.time()
print(end-start)
