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

    def forward(self, x714, x709):
        x715=operator.mul(x714, x709)
        return x715

m = M().eval()
x714 = torch.randn(torch.Size([1, 2304, 1, 1]))
x709 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x714, x709)
end = time.time()
print(end-start)
