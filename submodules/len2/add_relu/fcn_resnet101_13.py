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
        self.relu40 = ReLU(inplace=True)

    def forward(self, x150, x142):
        x151=operator.add(x150, x142)
        x152=self.relu40(x151)
        return x152

m = M().eval()
x150 = torch.randn(torch.Size([1, 1024, 28, 28]))
x142 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x150, x142)
end = time.time()
print(end-start)
