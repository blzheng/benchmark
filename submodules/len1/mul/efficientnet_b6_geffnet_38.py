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

    def forward(self, x567, x572):
        x573=operator.mul(x567, x572)
        return x573

m = M().eval()
x567 = torch.randn(torch.Size([1, 2064, 7, 7]))
x572 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x567, x572)
end = time.time()
print(end-start)
