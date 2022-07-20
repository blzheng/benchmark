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

    def forward(self, x284, x269, x300):
        x285=operator.add(x284, x269)
        x301=operator.add(x300, x285)
        return x301

m = M().eval()
x284 = torch.randn(torch.Size([1, 160, 14, 14]))
x269 = torch.randn(torch.Size([1, 160, 14, 14]))
x300 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x284, x269, x300)
end = time.time()
print(end-start)
