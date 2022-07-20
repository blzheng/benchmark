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
        self.conv2d109 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d112 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x389, x369, x376, x383, x397):
        x390=self.conv2d109(x389)
        x398=torch.cat([x369, x376, x383, x390, x397], 1)
        x399=self.batchnorm2d112(x398)
        return x399

m = M().eval()
x389 = torch.randn(torch.Size([1, 128, 7, 7]))
x369 = torch.randn(torch.Size([1, 640, 7, 7]))
x376 = torch.randn(torch.Size([1, 32, 7, 7]))
x383 = torch.randn(torch.Size([1, 32, 7, 7]))
x397 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x389, x369, x376, x383, x397)
end = time.time()
print(end-start)
