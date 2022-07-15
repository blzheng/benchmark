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

    def forward(self, x442, x437):
        x443=operator.mul(x442, x437)
        return x443

m = M().eval()
x442 = torch.randn(torch.Size([1, 960, 1, 1]))
x437 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x442, x437)
end = time.time()
print(end-start)
