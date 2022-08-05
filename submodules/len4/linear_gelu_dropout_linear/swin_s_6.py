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
        self.linear14 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)
        self.linear15 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x173):
        x174=self.linear14(x173)
        x175=self.gelu6(x174)
        x176=self.dropout12(x175)
        x177=self.linear15(x176)
        return x177

m = M().eval()
x173 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x173)
end = time.time()
print(end-start)
