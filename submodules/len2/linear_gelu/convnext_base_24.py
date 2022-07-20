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
        self.linear48 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu24 = GELU(approximate='none')

    def forward(self, x286):
        x287=self.linear48(x286)
        x288=self.gelu24(x287)
        return x288

m = M().eval()
x286 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x286)
end = time.time()
print(end-start)
