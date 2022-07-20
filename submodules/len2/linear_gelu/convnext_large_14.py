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
        self.linear28 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu14 = GELU(approximate='none')

    def forward(self, x176):
        x177=self.linear28(x176)
        x178=self.gelu14(x177)
        return x178

m = M().eval()
x176 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
