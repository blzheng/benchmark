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
        self.conv2d104 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x370, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238, x245, x252, x259, x266, x273, x280, x287, x294, x301, x308, x315, x322, x329, x336, x343, x350, x357, x364, x378):
        x371=self.conv2d104(x370)
        x379=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238, x245, x252, x259, x266, x273, x280, x287, x294, x301, x308, x315, x322, x329, x336, x343, x350, x357, x364, x371, x378], 1)
        x380=self.batchnorm2d107(x379)
        return x380

m = M().eval()
x370 = torch.randn(torch.Size([1, 192, 14, 14]))
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
x231 = torch.randn(torch.Size([1, 48, 14, 14]))
x238 = torch.randn(torch.Size([1, 48, 14, 14]))
x245 = torch.randn(torch.Size([1, 48, 14, 14]))
x252 = torch.randn(torch.Size([1, 48, 14, 14]))
x259 = torch.randn(torch.Size([1, 48, 14, 14]))
x266 = torch.randn(torch.Size([1, 48, 14, 14]))
x273 = torch.randn(torch.Size([1, 48, 14, 14]))
x280 = torch.randn(torch.Size([1, 48, 14, 14]))
x287 = torch.randn(torch.Size([1, 48, 14, 14]))
x294 = torch.randn(torch.Size([1, 48, 14, 14]))
x301 = torch.randn(torch.Size([1, 48, 14, 14]))
x308 = torch.randn(torch.Size([1, 48, 14, 14]))
x315 = torch.randn(torch.Size([1, 48, 14, 14]))
x322 = torch.randn(torch.Size([1, 48, 14, 14]))
x329 = torch.randn(torch.Size([1, 48, 14, 14]))
x336 = torch.randn(torch.Size([1, 48, 14, 14]))
x343 = torch.randn(torch.Size([1, 48, 14, 14]))
x350 = torch.randn(torch.Size([1, 48, 14, 14]))
x357 = torch.randn(torch.Size([1, 48, 14, 14]))
x364 = torch.randn(torch.Size([1, 48, 14, 14]))
x378 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x370, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231, x238, x245, x252, x259, x266, x273, x280, x287, x294, x301, x308, x315, x322, x329, x336, x343, x350, x357, x364, x378)
end = time.time()
print(end-start)
