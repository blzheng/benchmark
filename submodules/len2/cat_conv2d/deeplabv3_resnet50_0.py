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
        self.conv2d58 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x177, x180, x183, x186, x193):
        x194=torch.cat([x177, x180, x183, x186, x193],dim=1)
        x195=self.conv2d58(x194)
        return x195

m = M().eval()
x177 = torch.randn(torch.Size([1, 256, 28, 28]))
x180 = torch.randn(torch.Size([1, 256, 28, 28]))
x183 = torch.randn(torch.Size([1, 256, 28, 28]))
x186 = torch.randn(torch.Size([1, 256, 28, 28]))
x193 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x177, x180, x183, x186, x193)
end = time.time()
print(end-start)
