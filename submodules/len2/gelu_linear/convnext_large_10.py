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
        self.gelu10 = GELU(approximate='none')
        self.linear21 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x133):
        x134=self.gelu10(x133)
        x135=self.linear21(x134)
        return x135

m = M().eval()
x133 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
