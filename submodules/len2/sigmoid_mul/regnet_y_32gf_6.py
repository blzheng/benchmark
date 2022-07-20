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
        self.sigmoid6 = Sigmoid()

    def forward(self, x113, x109):
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        return x115

m = M().eval()
x113 = torch.randn(torch.Size([1, 696, 1, 1]))
x109 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x113, x109)
end = time.time()
print(end-start)
