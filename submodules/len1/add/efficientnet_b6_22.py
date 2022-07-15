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

    def forward(self, x431, x416):
        x432=operator.add(x431, x416)
        return x432

m = M().eval()
x431 = torch.randn(torch.Size([1, 200, 14, 14]))
x416 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x431, x416)
end = time.time()
print(end-start)
