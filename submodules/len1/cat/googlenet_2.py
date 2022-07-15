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

    def forward(self, x47, x53, x59, x63):
        x64=torch.cat([x47, x53, x59, x63], 1)
        return x64

m = M().eval()
x47 = torch.randn(torch.Size([1, 128, 28, 28]))
x53 = torch.randn(torch.Size([1, 192, 28, 28]))
x59 = torch.randn(torch.Size([1, 96, 28, 28]))
x63 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x47, x53, x59, x63)
end = time.time()
print(end-start)
