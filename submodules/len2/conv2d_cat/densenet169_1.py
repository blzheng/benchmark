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
        self.conv2d4 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x17, x4, x11, x25):
        x18=self.conv2d4(x17)
        x26=torch.cat([x4, x11, x18, x25], 1)
        return x26

m = M().eval()
x17 = torch.randn(torch.Size([1, 128, 56, 56]))
x4 = torch.randn(torch.Size([1, 64, 56, 56]))
x11 = torch.randn(torch.Size([1, 32, 56, 56]))
x25 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x17, x4, x11, x25)
end = time.time()
print(end-start)
