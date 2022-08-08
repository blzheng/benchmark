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

    def forward(self, x110, x107):
        x111=self.dropout6(x110)
        x112=operator.add(x111, x107)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 384, 256]))
x107 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x110, x107)
end = time.time()
print(end-start)
