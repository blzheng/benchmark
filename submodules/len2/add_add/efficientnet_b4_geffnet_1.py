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

    def forward(self, x130, x116, x145):
        x131=operator.add(x130, x116)
        x146=operator.add(x145, x131)
        return x146

m = M().eval()
x130 = torch.randn(torch.Size([1, 56, 28, 28]))
x116 = torch.randn(torch.Size([1, 56, 28, 28]))
x145 = torch.randn(torch.Size([1, 56, 28, 28]))
start = time.time()
output = m(x130, x116, x145)
end = time.time()
print(end-start)
