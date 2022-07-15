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

    def forward(self, x700, x705):
        x706=operator.mul(x700, x705)
        return x706

m = M().eval()
x700 = torch.randn(torch.Size([1, 2304, 7, 7]))
x705 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x700, x705)
end = time.time()
print(end-start)
