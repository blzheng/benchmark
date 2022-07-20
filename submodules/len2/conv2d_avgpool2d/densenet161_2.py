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
        self.conv2d111 = Conv2d(2112, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool2d2 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x395):
        x396=self.conv2d111(x395)
        x397=self.avgpool2d2(x396)
        return x397

m = M().eval()
x395 = torch.randn(torch.Size([1, 2112, 14, 14]))
start = time.time()
output = m(x395)
end = time.time()
print(end-start)
