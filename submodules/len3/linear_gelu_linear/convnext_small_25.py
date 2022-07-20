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
        self.linear50 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu25 = GELU(approximate='none')
        self.linear51 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x297):
        x298=self.linear50(x297)
        x299=self.gelu25(x298)
        x300=self.linear51(x299)
        return x300

m = M().eval()
x297 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)
