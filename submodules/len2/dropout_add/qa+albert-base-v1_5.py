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
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x243, x214):
        x244=self.dropout2(x243)
        x245=operator.add(x214, x244)
        return x245

m = M().eval()
x243 = torch.randn(torch.Size([1, 384, 768]))
x214 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x243, x214)
end = time.time()
print(end-start)
