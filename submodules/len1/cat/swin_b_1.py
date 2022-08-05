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

    def forward(self, x105, x106, x107, x108):
        x109=torch.cat([x105, x106, x107, x108], -1)
        return x109

m = M().eval()
x105 = torch.randn(torch.Size([1, 14, 14, 256]))
x106 = torch.randn(torch.Size([1, 14, 14, 256]))
x107 = torch.randn(torch.Size([1, 14, 14, 256]))
x108 = torch.randn(torch.Size([1, 14, 14, 256]))
start = time.time()
output = m(x105, x106, x107, x108)
end = time.time()
print(end-start)
