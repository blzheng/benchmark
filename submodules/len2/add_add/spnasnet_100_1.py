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

    def forward(self, x67, x58, x77):
        x68=operator.add(x67, x58)
        x78=operator.add(x77, x68)
        return x78

m = M().eval()
x67 = torch.randn(torch.Size([1, 40, 28, 28]))
x58 = torch.randn(torch.Size([1, 40, 28, 28]))
x77 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x67, x58, x77)
end = time.time()
print(end-start)
