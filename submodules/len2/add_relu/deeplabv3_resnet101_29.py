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
        self.relu88 = ReLU(inplace=True)

    def forward(self, x310, x302):
        x311=operator.add(x310, x302)
        x312=self.relu88(x311)
        return x312

m = M().eval()
x310 = torch.randn(torch.Size([1, 1024, 28, 28]))
x302 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x310, x302)
end = time.time()
print(end-start)
