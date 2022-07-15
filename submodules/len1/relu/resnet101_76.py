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
        self.relu76 = ReLU(inplace=True)

    def forward(self, x262):
        x263=self.relu76(x262)
        return x263

m = M().eval()
x262 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x262)
end = time.time()
print(end-start)
