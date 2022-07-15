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

    def forward(self, x256, x251):
        x257=operator.mul(x256, x251)
        return x257

m = M().eval()
x256 = torch.randn(torch.Size([1, 672, 1, 1]))
x251 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x256, x251)
end = time.time()
print(end-start)
