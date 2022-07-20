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

    def forward(self, x70, x61, x80):
        x71=operator.add(x70, x61)
        x81=operator.add(x80, x71)
        return x81

m = M().eval()
x70 = torch.randn(torch.Size([1, 32, 28, 28]))
x61 = torch.randn(torch.Size([1, 32, 28, 28]))
x80 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x70, x61, x80)
end = time.time()
print(end-start)
