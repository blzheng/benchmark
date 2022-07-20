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
        self.linear16 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu8 = GELU(approximate='none')

    def forward(self, x110):
        x111=self.linear16(x110)
        x112=self.gelu8(x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
