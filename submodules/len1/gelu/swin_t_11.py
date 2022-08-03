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
        self.gelu11 = GELU(approximate=none)

    def forward(self, x297):
        x298=self.gelu11(x297)
        return x298

m = M().eval()
x297 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)
