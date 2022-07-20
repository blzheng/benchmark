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
        self.gelu1 = GELU(approximate='none')

    def forward(self, x22):
        x23=self.gelu1(x22)
        return x23

m = M().eval()
x22 = torch.randn(torch.Size([1, 56, 56, 512]))
start = time.time()
output = m(x22)
end = time.time()
print(end-start)
