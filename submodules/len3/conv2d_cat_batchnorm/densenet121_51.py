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
        self.conv2d113 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x403, x313, x320, x327, x334, x341, x348, x355, x362, x369, x376, x383, x390, x397, x411):
        x404=self.conv2d113(x403)
        x412=torch.cat([x313, x320, x327, x334, x341, x348, x355, x362, x369, x376, x383, x390, x397, x404, x411], 1)
        x413=self.batchnorm2d116(x412)
        return x413

m = M().eval()
x403 = torch.randn(torch.Size([1, 128, 7, 7]))
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
x411 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x403, x313, x320, x327, x334, x341, x348, x355, x362, x369, x376, x383, x390, x397, x411)
end = time.time()
print(end-start)
