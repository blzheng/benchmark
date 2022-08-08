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

    def forward(self, x473, x461):
        x474=torch.matmul(x473, x461)
        return x474

m = M().eval()
x473 = torch.randn(torch.Size([1, 12, 384, 384]))
x461 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x473, x461)
end = time.time()
print(end-start)
