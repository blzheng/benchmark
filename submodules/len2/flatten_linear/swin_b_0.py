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
        self.linear51 = Linear(in_features=1024, out_features=1000, bias=True)

    def forward(self, x582):
        x583=torch.flatten(x582, 1)
        x584=self.linear51(x583)
        return x584

m = M().eval()
x582 = torch.randn(torch.Size([1, 1024, 1, 1]))
start = time.time()
output = m(x582)
end = time.time()
print(end-start)
