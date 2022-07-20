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

    def forward(self, x174, x160, x189):
        x175=operator.add(x174, x160)
        x190=operator.add(x189, x175)
        return x190

m = M().eval()
x174 = torch.randn(torch.Size([1, 96, 14, 14]))
x160 = torch.randn(torch.Size([1, 96, 14, 14]))
x189 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x174, x160, x189)
end = time.time()
print(end-start)
