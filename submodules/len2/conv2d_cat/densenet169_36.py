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
        self.conv2d80 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x286, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238, x245, x252, x259, x266, x273, x280, x294):
        x287=self.conv2d80(x286)
        x295=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238, x245, x252, x259, x266, x273, x280, x287, x294], 1)
        return x295

m = M().eval()
x286 = torch.randn(torch.Size([1, 128, 14, 14]))
x140 = torch.randn(torch.Size([1, 256, 14, 14]))
x147 = torch.randn(torch.Size([1, 32, 14, 14]))
x154 = torch.randn(torch.Size([1, 32, 14, 14]))
x161 = torch.randn(torch.Size([1, 32, 14, 14]))
x168 = torch.randn(torch.Size([1, 32, 14, 14]))
x175 = torch.randn(torch.Size([1, 32, 14, 14]))
x182 = torch.randn(torch.Size([1, 32, 14, 14]))
x189 = torch.randn(torch.Size([1, 32, 14, 14]))
x196 = torch.randn(torch.Size([1, 32, 14, 14]))
x203 = torch.randn(torch.Size([1, 32, 14, 14]))
x210 = torch.randn(torch.Size([1, 32, 14, 14]))
x217 = torch.randn(torch.Size([1, 32, 14, 14]))
x224 = torch.randn(torch.Size([1, 32, 14, 14]))
x231 = torch.randn(torch.Size([1, 32, 14, 14]))
x238 = torch.randn(torch.Size([1, 32, 14, 14]))
x245 = torch.randn(torch.Size([1, 32, 14, 14]))
x252 = torch.randn(torch.Size([1, 32, 14, 14]))
x259 = torch.randn(torch.Size([1, 32, 14, 14]))
x266 = torch.randn(torch.Size([1, 32, 14, 14]))
x273 = torch.randn(torch.Size([1, 32, 14, 14]))
x280 = torch.randn(torch.Size([1, 32, 14, 14]))
x294 = torch.randn(torch.Size([1, 32, 14, 14]))
start = time.time()
output = m(x286, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238, x245, x252, x259, x266, x273, x280, x294)
end = time.time()
print(end-start)
