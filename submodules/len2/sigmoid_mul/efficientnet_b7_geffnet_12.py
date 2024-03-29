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

    def forward(self, x182, x178):
        x183=x182.sigmoid()
        x184=operator.mul(x178, x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 480, 1, 1]))
x178 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x182, x178)
end = time.time()
print(end-start)
