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

    def forward(self, x277, x263, x292):
        x278=operator.add(x277, x263)
        x293=operator.add(x292, x278)
        return x293

m = M().eval()
x277 = torch.randn(torch.Size([1, 128, 14, 14]))
x263 = torch.randn(torch.Size([1, 128, 14, 14]))
x292 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x277, x263, x292)
end = time.time()
print(end-start)
