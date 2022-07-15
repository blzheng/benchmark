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

    def forward(self, x175, x170):
        x176=operator.mul(x175, x170)
        return x176

m = M().eval()
x175 = torch.randn(torch.Size([1, 384, 1, 1]))
x170 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x175, x170)
end = time.time()
print(end-start)
