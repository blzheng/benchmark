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
        self.conv2d64 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x230, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x238):
        x231=self.conv2d64(x230)
        x239=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238], 1)
        return x239

m = M().eval()
x230 = torch.randn(torch.Size([1, 192, 14, 14]))
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
x217 = torch.randn(torch.Size([1, 48, 14, 14]))
x224 = torch.randn(torch.Size([1, 48, 14, 14]))
x238 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x230, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x238)
end = time.time()
print(end-start)
