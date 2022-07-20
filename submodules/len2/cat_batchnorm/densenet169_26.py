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
        self.batchnorm2d51 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x140, x147, x154, x161, x168, x175, x182):
        x183=torch.cat([x140, x147, x154, x161, x168, x175, x182], 1)
        x184=self.batchnorm2d51(x183)
        return x184

m = M().eval()
x140 = torch.randn(torch.Size([1, 256, 14, 14]))
x147 = torch.randn(torch.Size([1, 32, 14, 14]))
x154 = torch.randn(torch.Size([1, 32, 14, 14]))
x161 = torch.randn(torch.Size([1, 32, 14, 14]))
x168 = torch.randn(torch.Size([1, 32, 14, 14]))
x175 = torch.randn(torch.Size([1, 32, 14, 14]))
x182 = torch.randn(torch.Size([1, 32, 14, 14]))
start = time.time()
output = m(x140, x147, x154, x161, x168, x175, x182)
end = time.time()
print(end-start)
