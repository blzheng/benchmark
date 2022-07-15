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

    def forward(self, x404, x409):
        x410=operator.mul(x404, x409)
        return x410

m = M().eval()
x404 = torch.randn(torch.Size([1, 1632, 7, 7]))
x409 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x404, x409)
end = time.time()
print(end-start)
