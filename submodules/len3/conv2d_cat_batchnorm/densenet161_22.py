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
        self.conv2d52 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x188, x140, x147, x154, x161, x168, x175, x182, x196):
        x189=self.conv2d52(x188)
        x197=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189, x196], 1)
        x198=self.batchnorm2d55(x197)
        return x198

m = M().eval()
x188 = torch.randn(torch.Size([1, 192, 14, 14]))
x140 = torch.randn(torch.Size([1, 384, 14, 14]))
x147 = torch.randn(torch.Size([1, 48, 14, 14]))
x154 = torch.randn(torch.Size([1, 48, 14, 14]))
x161 = torch.randn(torch.Size([1, 48, 14, 14]))
x168 = torch.randn(torch.Size([1, 48, 14, 14]))
x175 = torch.randn(torch.Size([1, 48, 14, 14]))
x182 = torch.randn(torch.Size([1, 48, 14, 14]))
x196 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x188, x140, x147, x154, x161, x168, x175, x182, x196)
end = time.time()
print(end-start)
