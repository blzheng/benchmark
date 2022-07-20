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

    def forward(self, x571, x567):
        x572=x571.sigmoid()
        x573=operator.mul(x567, x572)
        return x573

m = M().eval()
x571 = torch.randn(torch.Size([1, 2064, 1, 1]))
x567 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x571, x567)
end = time.time()
print(end-start)
