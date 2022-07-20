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
        self.gelu8 = GELU(approximate='none')
        self.linear17 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x111):
        x112=self.gelu8(x111)
        x113=self.linear17(x112)
        return x113

m = M().eval()
x111 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)
