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

    def forward(self, x159, x145):
        x160=operator.add(x159, x145)
        return x160

m = M().eval()
x159 = torch.randn(torch.Size([1, 96, 14, 14]))
x145 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x159, x145)
end = time.time()
print(end-start)
