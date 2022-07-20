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
        self.conv2d127 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d130 = BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x452, x397, x404, x411, x418, x425, x432, x439, x446, x460):
        x453=self.conv2d127(x452)
        x461=torch.cat([x397, x404, x411, x418, x425, x432, x439, x446, x453, x460], 1)
        x462=self.batchnorm2d130(x461)
        return x462

m = M().eval()
x452 = torch.randn(torch.Size([1, 192, 7, 7]))
x397 = torch.randn(torch.Size([1, 1056, 7, 7]))
x404 = torch.randn(torch.Size([1, 48, 7, 7]))
x411 = torch.randn(torch.Size([1, 48, 7, 7]))
x418 = torch.randn(torch.Size([1, 48, 7, 7]))
x425 = torch.randn(torch.Size([1, 48, 7, 7]))
x432 = torch.randn(torch.Size([1, 48, 7, 7]))
x439 = torch.randn(torch.Size([1, 48, 7, 7]))
x446 = torch.randn(torch.Size([1, 48, 7, 7]))
x460 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x452, x397, x404, x411, x418, x425, x432, x439, x446, x460)
end = time.time()
print(end-start)
