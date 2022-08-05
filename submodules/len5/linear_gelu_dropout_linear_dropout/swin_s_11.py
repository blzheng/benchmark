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
        self.linear24 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear25 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)

    def forward(self, x288):
        x289=self.linear24(x288)
        x290=self.gelu11(x289)
        x291=self.dropout22(x290)
        x292=self.linear25(x291)
        x293=self.dropout23(x292)
        return x293

m = M().eval()
x288 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x288)
end = time.time()
print(end-start)
