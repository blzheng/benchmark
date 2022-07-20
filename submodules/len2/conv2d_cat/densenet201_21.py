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
        self.conv2d50 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x181, x140, x147, x154, x161, x168, x175, x189):
        x182=self.conv2d50(x181)
        x190=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189], 1)
        return x190

m = M().eval()
x181 = torch.randn(torch.Size([1, 128, 14, 14]))
x140 = torch.randn(torch.Size([1, 256, 14, 14]))
x147 = torch.randn(torch.Size([1, 32, 14, 14]))
x154 = torch.randn(torch.Size([1, 32, 14, 14]))
x161 = torch.randn(torch.Size([1, 32, 14, 14]))
x168 = torch.randn(torch.Size([1, 32, 14, 14]))
x175 = torch.randn(torch.Size([1, 32, 14, 14]))
x189 = torch.randn(torch.Size([1, 32, 14, 14]))
start = time.time()
output = m(x181, x140, x147, x154, x161, x168, x175, x189)
end = time.time()
print(end-start)
