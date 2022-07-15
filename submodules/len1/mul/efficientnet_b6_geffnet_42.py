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

    def forward(self, x627, x632):
        x633=operator.mul(x627, x632)
        return x633

m = M().eval()
x627 = torch.randn(torch.Size([1, 2064, 7, 7]))
x632 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x627, x632)
end = time.time()
print(end-start)
