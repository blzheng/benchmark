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
        self.linear52 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu26 = GELU(approximate='none')

    def forward(self, x308):
        x309=self.linear52(x308)
        x310=self.gelu26(x309)
        return x310

m = M().eval()
x308 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x308)
end = time.time()
print(end-start)
