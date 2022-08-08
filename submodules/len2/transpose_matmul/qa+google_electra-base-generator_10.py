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

    def forward(self, x456, x467):
        x468=x456.transpose(-1, -2)
        x469=torch.matmul(x467, x468)
        return x469

m = M().eval()
x456 = torch.randn(torch.Size([1, 4, 384, 64]))
x467 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x456, x467)
end = time.time()
print(end-start)
