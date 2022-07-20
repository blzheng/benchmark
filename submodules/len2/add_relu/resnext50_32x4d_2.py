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
        self.relu7 = ReLU(inplace=True)

    def forward(self, x34, x26):
        x35=operator.add(x34, x26)
        x36=self.relu7(x35)
        return x36

m = M().eval()
x34 = torch.randn(torch.Size([1, 256, 56, 56]))
x26 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x34, x26)
end = time.time()
print(end-start)
