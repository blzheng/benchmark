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
        self.relu9 = ReLU(inplace=True)

    def forward(self, x39, x34):
        x40=operator.add(x39, x34)
        x41=self.relu9(x40)
        return x41

m = M().eval()
x39 = torch.randn(torch.Size([1, 128, 28, 28]))
x34 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x39, x34)
end = time.time()
print(end-start)
