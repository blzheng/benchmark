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

    def forward(self, x285, x290):
        x291=operator.mul(x285, x290)
        return x291

m = M().eval()
x285 = torch.randn(torch.Size([1, 960, 14, 14]))
x290 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x285, x290)
end = time.time()
print(end-start)
