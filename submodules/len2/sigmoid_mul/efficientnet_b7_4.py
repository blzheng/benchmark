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
        self.sigmoid4 = Sigmoid()

    def forward(self, x63, x59):
        x64=self.sigmoid4(x63)
        x65=operator.mul(x64, x59)
        return x65

m = M().eval()
x63 = torch.randn(torch.Size([1, 192, 1, 1]))
x59 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x63, x59)
end = time.time()
print(end-start)
