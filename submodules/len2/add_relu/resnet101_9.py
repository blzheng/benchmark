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
        self.relu28 = ReLU(inplace=True)

    def forward(self, x108, x100):
        x109=operator.add(x108, x100)
        x110=self.relu28(x109)
        return x110

m = M().eval()
x108 = torch.randn(torch.Size([1, 1024, 14, 14]))
x100 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x108, x100)
end = time.time()
print(end-start)
