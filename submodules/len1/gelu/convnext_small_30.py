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
        self.gelu30 = GELU(approximate='none')

    def forward(self, x353):
        x354=self.gelu30(x353)
        return x354

m = M().eval()
x353 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x353)
end = time.time()
print(end-start)
