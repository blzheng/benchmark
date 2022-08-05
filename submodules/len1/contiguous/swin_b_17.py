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

    def forward(self, x416):
        x417=x416.contiguous()
        return x417

m = M().eval()
x416 = torch.randn(torch.Size([16, 49, 49]))
start = time.time()
output = m(x416)
end = time.time()
print(end-start)
