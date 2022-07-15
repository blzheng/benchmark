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

    def forward(self, x582, x587):
        x588=operator.mul(x582, x587)
        return x588

m = M().eval()
x582 = torch.randn(torch.Size([1, 2064, 7, 7]))
x587 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x582, x587)
end = time.time()
print(end-start)
