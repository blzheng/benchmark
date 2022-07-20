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
        self.conv2d38 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()

    def forward(self, x113, x110):
        x114=self.conv2d38(x113)
        x115=self.sigmoid7(x114)
        x116=operator.mul(x115, x110)
        return x116

m = M().eval()
x113 = torch.randn(torch.Size([1, 20, 1, 1]))
x110 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x113, x110)
end = time.time()
print(end-start)
