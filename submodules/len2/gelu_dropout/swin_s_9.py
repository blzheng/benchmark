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
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)

    def forward(self, x243):
        x244=self.gelu9(x243)
        x245=self.dropout18(x244)
        return x245

m = M().eval()
x243 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x243)
end = time.time()
print(end-start)