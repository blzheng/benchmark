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

    def forward(self, x33, x41, x50):
        x42=operator.add(x33, x41)
        x51=operator.add(x42, x50)
        return x51

m = M().eval()
x33 = torch.randn(torch.Size([1, 32, 28, 28]))
x41 = torch.randn(torch.Size([1, 32, 28, 28]))
x50 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x33, x41, x50)
end = time.time()
print(end-start)
