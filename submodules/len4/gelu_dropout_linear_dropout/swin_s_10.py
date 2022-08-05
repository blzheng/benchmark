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
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear23 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)

    def forward(self, x266):
        x267=self.gelu10(x266)
        x268=self.dropout20(x267)
        x269=self.linear23(x268)
        x270=self.dropout21(x269)
        return x270

m = M().eval()
x266 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x266)
end = time.time()
print(end-start)
