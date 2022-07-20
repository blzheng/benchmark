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

    def forward(self, x180, x165, x196):
        x181=operator.add(x180, x165)
        x197=operator.add(x196, x181)
        return x197

m = M().eval()
x180 = torch.randn(torch.Size([1, 96, 14, 14]))
x165 = torch.randn(torch.Size([1, 96, 14, 14]))
x196 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x180, x165, x196)
end = time.time()
print(end-start)
