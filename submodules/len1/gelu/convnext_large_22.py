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
        self.gelu22 = GELU(approximate='none')

    def forward(self, x265):
        x266=self.gelu22(x265)
        return x266

m = M().eval()
x265 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x265)
end = time.time()
print(end-start)
