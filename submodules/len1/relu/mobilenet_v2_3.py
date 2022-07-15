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
        self.relu63 = ReLU6(inplace=True)

    def forward(self, x13):
        x14=self.relu63(x13)
        return x14

m = M().eval()
x13 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
