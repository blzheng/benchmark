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
        self.conv2d60 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x216, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x224):
        x217=self.conv2d60(x216)
        x225=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224], 1)
        return x225

m = M().eval()
x216 = torch.randn(torch.Size([1, 192, 14, 14]))
x140 = torch.randn(torch.Size([1, 384, 14, 14]))
x147 = torch.randn(torch.Size([1, 48, 14, 14]))
x154 = torch.randn(torch.Size([1, 48, 14, 14]))
x161 = torch.randn(torch.Size([1, 48, 14, 14]))
x168 = torch.randn(torch.Size([1, 48, 14, 14]))
x175 = torch.randn(torch.Size([1, 48, 14, 14]))
x182 = torch.randn(torch.Size([1, 48, 14, 14]))
x189 = torch.randn(torch.Size([1, 48, 14, 14]))
x196 = torch.randn(torch.Size([1, 48, 14, 14]))
x203 = torch.randn(torch.Size([1, 48, 14, 14]))
x210 = torch.randn(torch.Size([1, 48, 14, 14]))
x224 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x216, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x224)
end = time.time()
print(end-start)
