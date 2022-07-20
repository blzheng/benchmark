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

    def forward(self, x369, x359, x380):
        x370=operator.add(x369, x359)
        x381=operator.add(x380, x370)
        return x381

m = M().eval()
x369 = torch.randn(torch.Size([1, 768, 14, 14]))
x359 = torch.randn(torch.Size([1, 768, 14, 14]))
x380 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x369, x359, x380)
end = time.time()
print(end-start)
