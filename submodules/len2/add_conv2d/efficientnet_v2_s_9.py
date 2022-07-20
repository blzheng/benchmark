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
        self.conv2d34 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x110, x95):
        x111=operator.add(x110, x95)
        x112=self.conv2d34(x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 128, 14, 14]))
x95 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x110, x95)
end = time.time()
print(end-start)
