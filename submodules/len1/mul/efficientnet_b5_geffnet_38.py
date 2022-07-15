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

    def forward(self, x566, x571):
        x572=operator.mul(x566, x571)
        return x572

m = M().eval()
x566 = torch.randn(torch.Size([1, 3072, 7, 7]))
x571 = torch.randn(torch.Size([1, 3072, 1, 1]))
start = time.time()
output = m(x566, x571)
end = time.time()
print(end-start)
