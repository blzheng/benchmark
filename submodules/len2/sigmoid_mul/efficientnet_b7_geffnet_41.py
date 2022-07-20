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

    def forward(self, x614, x610):
        x615=x614.sigmoid()
        x616=operator.mul(x610, x615)
        return x616

m = M().eval()
x614 = torch.randn(torch.Size([1, 2304, 1, 1]))
x610 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x614, x610)
end = time.time()
print(end-start)
