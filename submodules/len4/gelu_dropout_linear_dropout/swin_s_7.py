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
        self.gelu7 = GELU(approximate='none')
        self.dropout14 = Dropout(p=0.0, inplace=False)
        self.linear17 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout15 = Dropout(p=0.0, inplace=False)

    def forward(self, x197):
        x198=self.gelu7(x197)
        x199=self.dropout14(x198)
        x200=self.linear17(x199)
        x201=self.dropout15(x200)
        return x201

m = M().eval()
x197 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x197)
end = time.time()
print(end-start)
