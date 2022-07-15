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

    def forward(self, x774, x769):
        x775=operator.mul(x774, x769)
        return x775

m = M().eval()
x774 = torch.randn(torch.Size([1, 2304, 1, 1]))
x769 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x774, x769)
end = time.time()
print(end-start)
