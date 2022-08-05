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

    def forward(self, x195):
        x196=x195.mean((2, 3),keepdim=True)
        return x196

m = M().eval()
x195 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x195)
end = time.time()
print(end-start)
