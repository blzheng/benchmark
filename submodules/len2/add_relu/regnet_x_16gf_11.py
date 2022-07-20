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
        self.relu36 = ReLU(inplace=True)

    def forward(self, x119, x127):
        x128=operator.add(x119, x127)
        x129=self.relu36(x128)
        return x129

m = M().eval()
x119 = torch.randn(torch.Size([1, 896, 14, 14]))
x127 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x119, x127)
end = time.time()
print(end-start)
