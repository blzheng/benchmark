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

    def forward(self, x220, x215):
        x221=operator.mul(x220, x215)
        return x221

m = M().eval()
x220 = torch.randn(torch.Size([1, 480, 1, 1]))
x215 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x220, x215)
end = time.time()
print(end-start)
