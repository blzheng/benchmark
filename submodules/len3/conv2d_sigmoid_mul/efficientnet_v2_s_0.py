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
        self.conv2d22 = Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()

    def forward(self, x74, x71):
        x75=self.conv2d22(x74)
        x76=self.sigmoid0(x75)
        x77=operator.mul(x76, x71)
        return x77

m = M().eval()
x74 = torch.randn(torch.Size([1, 16, 1, 1]))
x71 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x74, x71)
end = time.time()
print(end-start)
