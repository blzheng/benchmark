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

    def forward(self, x41, x35):
        x42=operator.add(x41, x35)
        return x42

m = M().eval()
x41 = torch.randn(torch.Size([1, 64, 56, 56]))
x35 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x41, x35)
end = time.time()
print(end-start)
