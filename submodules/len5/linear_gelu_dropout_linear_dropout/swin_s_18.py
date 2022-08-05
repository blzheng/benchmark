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
        self.linear38 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu18 = GELU(approximate='none')
        self.dropout36 = Dropout(p=0.0, inplace=False)
        self.linear39 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout37 = Dropout(p=0.0, inplace=False)

    def forward(self, x449):
        x450=self.linear38(x449)
        x451=self.gelu18(x450)
        x452=self.dropout36(x451)
        x453=self.linear39(x452)
        x454=self.dropout37(x453)
        return x454

m = M().eval()
x449 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x449)
end = time.time()
print(end-start)
