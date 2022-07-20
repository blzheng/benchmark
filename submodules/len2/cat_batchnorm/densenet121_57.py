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
        self.batchnorm2d112 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x313, x320, x327, x334, x341, x348, x355, x362, x369, x376, x383, x390, x397):
        x398=torch.cat([x313, x320, x327, x334, x341, x348, x355, x362, x369, x376, x383, x390, x397], 1)
        x399=self.batchnorm2d112(x398)
        return x399

m = M().eval()
x313 = torch.randn(torch.Size([1, 512, 7, 7]))
x320 = torch.randn(torch.Size([1, 32, 7, 7]))
x327 = torch.randn(torch.Size([1, 32, 7, 7]))
x334 = torch.randn(torch.Size([1, 32, 7, 7]))
x341 = torch.randn(torch.Size([1, 32, 7, 7]))
x348 = torch.randn(torch.Size([1, 32, 7, 7]))
x355 = torch.randn(torch.Size([1, 32, 7, 7]))
x362 = torch.randn(torch.Size([1, 32, 7, 7]))
x369 = torch.randn(torch.Size([1, 32, 7, 7]))
x376 = torch.randn(torch.Size([1, 32, 7, 7]))
x383 = torch.randn(torch.Size([1, 32, 7, 7]))
x390 = torch.randn(torch.Size([1, 32, 7, 7]))
x397 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x313, x320, x327, x334, x341, x348, x355, x362, x369, x376, x383, x390, x397)
end = time.time()
print(end-start)
