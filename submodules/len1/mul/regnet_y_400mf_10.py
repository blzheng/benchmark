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

    def forward(self, x182, x177):
        x183=operator.mul(x182, x177)
        return x183

m = M().eval()
x182 = torch.randn(torch.Size([1, 440, 1, 1]))
x177 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x182, x177)
end = time.time()
print(end-start)
