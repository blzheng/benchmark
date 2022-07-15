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

    def forward(self, x114, x100):
        x115=operator.add(x114, x100)
        return x115

m = M().eval()
x114 = torch.randn(torch.Size([1, 40, 56, 56]))
x100 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x114, x100)
end = time.time()
print(end-start)
