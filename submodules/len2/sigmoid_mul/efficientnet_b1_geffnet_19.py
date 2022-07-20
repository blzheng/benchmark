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

    def forward(self, x288, x284):
        x289=x288.sigmoid()
        x290=operator.mul(x284, x289)
        return x290

m = M().eval()
x288 = torch.randn(torch.Size([1, 1152, 1, 1]))
x284 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x288, x284)
end = time.time()
print(end-start)
