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

    def forward(self, x649, x635, x664):
        x650=operator.add(x649, x635)
        x665=operator.add(x664, x650)
        return x665

m = M().eval()
x649 = torch.randn(torch.Size([1, 576, 7, 7]))
x635 = torch.randn(torch.Size([1, 576, 7, 7]))
x664 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x649, x635, x664)
end = time.time()
print(end-start)
