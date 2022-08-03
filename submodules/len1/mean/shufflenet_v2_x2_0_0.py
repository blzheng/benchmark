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

    def forward(self, x365):
        x366=x365.mean([2, 3])
        return x366

m = M().eval()
x365 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x365)
end = time.time()
print(end-start)
