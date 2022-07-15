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

    def forward(self, x256, x241):
        x257=operator.add(x256, x241)
        return x257

m = M().eval()
x256 = torch.randn(torch.Size([1, 80, 28, 28]))
x241 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x256, x241)
end = time.time()
print(end-start)
