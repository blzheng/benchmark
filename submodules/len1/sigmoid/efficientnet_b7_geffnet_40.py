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

    def forward(self, x599):
        x600=x599.sigmoid()
        return x600

m = M().eval()
x599 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x599)
end = time.time()
print(end-start)
