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
        self.relu22 = ReLU(inplace=True)

    def forward(self, x86, x88):
        x89=operator.add(x86, x88)
        x90=self.relu22(x89)
        return x90

m = M().eval()
x86 = torch.randn(torch.Size([1, 1024, 14, 14]))
x88 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x86, x88)
end = time.time()
print(end-start)
