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
        self.linear22 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)

    def forward(self, x265):
        x266=self.linear22(x265)
        x267=self.gelu10(x266)
        x268=self.dropout20(x267)
        return x268

m = M().eval()
x265 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x265)
end = time.time()
print(end-start)
