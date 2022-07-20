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

    def forward(self, x22, x17):
        x23=operator.mul(x22, x17)
        return x23

m = M().eval()
x22 = torch.randn(torch.Size([1, 16, 1, 1]))
x17 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x22, x17)
end = time.time()
print(end-start)