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

    def forward(self, x727, x722):
        x728=operator.mul(x727, x722)
        return x728

m = M().eval()
x727 = torch.randn(torch.Size([1, 3072, 1, 1]))
x722 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x727, x722)
end = time.time()
print(end-start)
