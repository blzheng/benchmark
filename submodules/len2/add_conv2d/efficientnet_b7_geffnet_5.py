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
        self.conv2d37 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x112, x98):
        x113=operator.add(x112, x98)
        x114=self.conv2d37(x113)
        return x114

m = M().eval()
x112 = torch.randn(torch.Size([1, 48, 56, 56]))
x98 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x112, x98)
end = time.time()
print(end-start)
