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
        self.relu106 = ReLU(inplace=True)

    def forward(self, x432):
        x433=self.relu106(x432)
        return x433

m = M().eval()
x432 = torch.randn(torch.Size([1, 7392, 7, 7]))
start = time.time()
output = m(x432)
end = time.time()
print(end-start)
