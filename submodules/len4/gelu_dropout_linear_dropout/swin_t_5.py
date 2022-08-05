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
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)
        self.linear13 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout11 = Dropout(p=0.0, inplace=False)

    def forward(self, x151):
        x152=self.gelu5(x151)
        x153=self.dropout10(x152)
        x154=self.linear13(x153)
        x155=self.dropout11(x154)
        return x155

m = M().eval()
x151 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x151)
end = time.time()
print(end-start)
