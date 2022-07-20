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
        self.relu91 = ReLU(inplace=True)

    def forward(self, x319):
        x320=self.relu91(x319)
        return x320

m = M().eval()
x319 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x319)
end = time.time()
print(end-start)