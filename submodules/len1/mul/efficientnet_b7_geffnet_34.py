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

    def forward(self, x506, x511):
        x512=operator.mul(x506, x511)
        return x512

m = M().eval()
x506 = torch.randn(torch.Size([1, 1344, 14, 14]))
x511 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x506, x511)
end = time.time()
print(end-start)
