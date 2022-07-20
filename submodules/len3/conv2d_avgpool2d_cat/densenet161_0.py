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
        self.conv2d13 = Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool2d0 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x49, x58):
        x50=self.conv2d13(x49)
        x51=self.avgpool2d0(x50)
        x59=torch.cat([x51, x58], 1)
        return x59

m = M().eval()
x49 = torch.randn(torch.Size([1, 384, 56, 56]))
x58 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x49, x58)
end = time.time()
print(end-start)
