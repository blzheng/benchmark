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

    def forward(self, x180, x175):
        x181=operator.mul(x180, x175)
        return x181

m = M().eval()
x180 = torch.randn(torch.Size([1, 320, 1, 1]))
x175 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x180, x175)
end = time.time()
print(end-start)
