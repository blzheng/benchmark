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

    def forward(self, x50, x42, x59):
        x51=operator.add(x50, x42)
        x60=operator.add(x59, x51)
        return x60

m = M().eval()
x50 = torch.randn(torch.Size([1, 40, 28, 28]))
x42 = torch.randn(torch.Size([1, 40, 28, 28]))
x59 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x50, x42, x59)
end = time.time()
print(end-start)
