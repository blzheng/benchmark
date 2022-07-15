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

    def forward(self, x842, x837):
        x843=operator.mul(x842, x837)
        return x843

m = M().eval()
x842 = torch.randn(torch.Size([1, 2304, 1, 1]))
x837 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x842, x837)
end = time.time()
print(end-start)
