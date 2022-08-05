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

    def forward(self, x274):
        x275=self.gelu10(x274)
        x276=self.dropout20(x275)
        return x276

m = M().eval()
x274 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
