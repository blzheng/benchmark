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
        self.linear0 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)

    def forward(self, x19):
        x20=self.linear0(x19)
        x21=self.gelu0(x20)
        x22=self.dropout0(x21)
        return x22

m = M().eval()
x19 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
