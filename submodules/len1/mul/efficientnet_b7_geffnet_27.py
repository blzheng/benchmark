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

    def forward(self, x402, x407):
        x408=operator.mul(x402, x407)
        return x408

m = M().eval()
x402 = torch.randn(torch.Size([1, 960, 14, 14]))
x407 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x402, x407)
end = time.time()
print(end-start)
