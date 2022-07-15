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

    def forward(self, x149, x139):
        x150=operator.add(x149, x139)
        return x150

m = M().eval()
x149 = torch.randn(torch.Size([1, 768, 14, 14]))
x139 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x149, x139)
end = time.time()
print(end-start)
