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
        self.dropout30 = Dropout(p=0.0, inplace=False)
        self.linear33 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout31 = Dropout(p=0.0, inplace=False)

    def forward(self, x382):
        x383=self.dropout30(x382)
        x384=self.linear33(x383)
        x385=self.dropout31(x384)
        return x385

m = M().eval()
x382 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x382)
end = time.time()
print(end-start)
