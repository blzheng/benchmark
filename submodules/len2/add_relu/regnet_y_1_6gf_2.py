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
        self.relu12 = ReLU(inplace=True)

    def forward(self, x39, x53):
        x54=operator.add(x39, x53)
        x55=self.relu12(x54)
        return x55

m = M().eval()
x39 = torch.randn(torch.Size([1, 120, 28, 28]))
x53 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x39, x53)
end = time.time()
print(end-start)
