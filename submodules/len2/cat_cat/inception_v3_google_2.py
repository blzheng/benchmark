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

    def forward(self, x301, x304, x295, x318, x322):
        x305=torch.cat([x301, x304], 1)
        x323=torch.cat([x295, x305, x318, x322], 1)
        return x323

m = M().eval()
x301 = torch.randn(torch.Size([1, 384, 5, 5]))
x304 = torch.randn(torch.Size([1, 384, 5, 5]))
x295 = torch.randn(torch.Size([1, 320, 5, 5]))
x318 = torch.randn(torch.Size([1, 768, 5, 5]))
x322 = torch.randn(torch.Size([1, 192, 5, 5]))
start = time.time()
output = m(x301, x304, x295, x318, x322)
end = time.time()
print(end-start)
