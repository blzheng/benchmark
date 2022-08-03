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
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear23 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x267):
        x268=self.dropout20(x267)
        x269=self.linear23(x268)
        return x269

m = M().eval()
x267 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x267)
end = time.time()
print(end-start)
