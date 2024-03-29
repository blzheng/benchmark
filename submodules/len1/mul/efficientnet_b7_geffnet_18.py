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

    def forward(self, x268, x273):
        x274=operator.mul(x268, x273)
        return x274

m = M().eval()
x268 = torch.randn(torch.Size([1, 480, 14, 14]))
x273 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x268, x273)
end = time.time()
print(end-start)
