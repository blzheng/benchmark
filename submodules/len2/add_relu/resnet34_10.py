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
        self.relu21 = ReLU(inplace=True)

    def forward(self, x83, x78):
        x84=operator.add(x83, x78)
        x85=self.relu21(x84)
        return x85

m = M().eval()
x83 = torch.randn(torch.Size([1, 256, 14, 14]))
x78 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x83, x78)
end = time.time()
print(end-start)
