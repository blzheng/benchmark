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
        self.dropout6 = Dropout(p=0.1, inplace=False)

    def forward(self, x109, x106):
        x110=self.dropout6(x109)
        x111=operator.add(x110, x106)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 384, 768]))
x106 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x109, x106)
end = time.time()
print(end-start)
