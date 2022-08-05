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

    def forward(self, x115):
        x116=x115.view(49, 49, -1)
        return x116

m = M().eval()
x115 = torch.randn(torch.Size([2401, 16]))
start = time.time()
output = m(x115)
end = time.time()
print(end-start)
