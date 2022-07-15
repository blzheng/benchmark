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

    def forward(self, x54, x40):
        x55=operator.add(x54, x40)
        return x55

m = M().eval()
x54 = torch.randn(torch.Size([1, 40, 28, 28]))
x40 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x54, x40)
end = time.time()
print(end-start)
