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
        self.linear41 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x244):
        x245=self.linear41(x244)
        x246=torch.permute(x245, [0, 3, 1, 2])
        return x246

m = M().eval()
x244 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x244)
end = time.time()
print(end-start)