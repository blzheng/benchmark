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

    def forward(self, x99, x94):
        x100=operator.mul(x99, x94)
        return x100

m = M().eval()
x99 = torch.randn(torch.Size([1, 480, 1, 1]))
x94 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x99, x94)
end = time.time()
print(end-start)