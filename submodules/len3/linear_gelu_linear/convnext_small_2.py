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
        self.linear4 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.linear5 = Linear(in_features=384, out_features=96, bias=True)

    def forward(self, x32):
        x33=self.linear4(x32)
        x34=self.gelu2(x33)
        x35=self.linear5(x34)
        return x35

m = M().eval()
x32 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
