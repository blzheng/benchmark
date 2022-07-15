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

    def forward(self, x463, x468):
        x469=operator.mul(x463, x468)
        return x469

m = M().eval()
x463 = torch.randn(torch.Size([1, 2688, 7, 7]))
x468 = torch.randn(torch.Size([1, 2688, 1, 1]))
start = time.time()
output = m(x463, x468)
end = time.time()
print(end-start)
