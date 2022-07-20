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
        self.conv2d121 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d124 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x431, x369, x376, x383, x390, x397, x404, x411, x418, x425, x439):
        x432=self.conv2d121(x431)
        x440=torch.cat([x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439], 1)
        x441=self.batchnorm2d124(x440)
        return x441

m = M().eval()
x431 = torch.randn(torch.Size([1, 128, 7, 7]))
x369 = torch.randn(torch.Size([1, 640, 7, 7]))
x376 = torch.randn(torch.Size([1, 32, 7, 7]))
x383 = torch.randn(torch.Size([1, 32, 7, 7]))
x390 = torch.randn(torch.Size([1, 32, 7, 7]))
x397 = torch.randn(torch.Size([1, 32, 7, 7]))
x404 = torch.randn(torch.Size([1, 32, 7, 7]))
x411 = torch.randn(torch.Size([1, 32, 7, 7]))
x418 = torch.randn(torch.Size([1, 32, 7, 7]))
x425 = torch.randn(torch.Size([1, 32, 7, 7]))
x439 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x431, x369, x376, x383, x390, x397, x404, x411, x418, x425, x439)
end = time.time()
print(end-start)
