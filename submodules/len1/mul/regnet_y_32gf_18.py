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

    def forward(self, x308, x303):
        x309=operator.mul(x308, x303)
        return x309

m = M().eval()
x308 = torch.randn(torch.Size([1, 1392, 1, 1]))
x303 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x308, x303)
end = time.time()
print(end-start)
