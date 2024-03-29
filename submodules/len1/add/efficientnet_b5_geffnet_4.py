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

    def forward(self, x99, x85):
        x100=operator.add(x99, x85)
        return x100

m = M().eval()
x99 = torch.randn(torch.Size([1, 40, 56, 56]))
x85 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x99, x85)
end = time.time()
print(end-start)
