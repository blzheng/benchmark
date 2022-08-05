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
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)
        self.linear15 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x174):
        x175=self.gelu6(x174)
        x176=self.dropout12(x175)
        x177=self.linear15(x176)
        return x177

m = M().eval()
x174 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
