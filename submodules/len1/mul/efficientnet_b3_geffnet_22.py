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

    def forward(self, x329, x334):
        x335=operator.mul(x329, x334)
        return x335

m = M().eval()
x329 = torch.randn(torch.Size([1, 1392, 7, 7]))
x334 = torch.randn(torch.Size([1, 1392, 1, 1]))
start = time.time()
output = m(x329, x334)
end = time.time()
print(end-start)
