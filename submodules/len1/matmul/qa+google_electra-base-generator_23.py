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

    def forward(self, x516, x504):
        x517=torch.matmul(x516, x504)
        return x517

m = M().eval()
x516 = torch.randn(torch.Size([1, 4, 384, 384]))
x504 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x516, x504)
end = time.time()
print(end-start)
