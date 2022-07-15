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

    def forward(self, x221, x216):
        x222=operator.mul(x221, x216)
        return x222

m = M().eval()
x221 = torch.randn(torch.Size([1, 768, 1, 1]))
x216 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x221, x216)
end = time.time()
print(end-start)
