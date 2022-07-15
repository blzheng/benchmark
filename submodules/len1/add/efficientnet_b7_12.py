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

    def forward(self, x240, x225):
        x241=operator.add(x240, x225)
        return x241

m = M().eval()
x240 = torch.randn(torch.Size([1, 80, 28, 28]))
x225 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x240, x225)
end = time.time()
print(end-start)
