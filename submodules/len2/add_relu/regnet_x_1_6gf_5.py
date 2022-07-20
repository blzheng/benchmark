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
        self.relu18 = ReLU(inplace=True)

    def forward(self, x57, x65):
        x66=operator.add(x57, x65)
        x67=self.relu18(x66)
        return x67

m = M().eval()
x57 = torch.randn(torch.Size([1, 168, 28, 28]))
x65 = torch.randn(torch.Size([1, 168, 28, 28]))
start = time.time()
output = m(x57, x65)
end = time.time()
print(end-start)
