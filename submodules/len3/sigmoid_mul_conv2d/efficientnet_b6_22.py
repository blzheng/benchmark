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
        self.sigmoid22 = Sigmoid()
        self.conv2d112 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x348, x344):
        x349=self.sigmoid22(x348)
        x350=operator.mul(x349, x344)
        x351=self.conv2d112(x350)
        return x351

m = M().eval()
x348 = torch.randn(torch.Size([1, 864, 1, 1]))
x344 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x348, x344)
end = time.time()
print(end-start)
