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
        self.linear35 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x275):
        x276=self.linear35(x275)
        x277=torch._C._nn.gelu(x276)
        return x277

m = M().eval()
x275 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
