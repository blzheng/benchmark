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
        self.relu97 = ReLU(inplace=True)

    def forward(self, x338, x330):
        x339=operator.add(x338, x330)
        x340=self.relu97(x339)
        return x340

m = M().eval()
x338 = torch.randn(torch.Size([1, 1024, 14, 14]))
x330 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x338, x330)
end = time.time()
print(end-start)
