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
        self.conv2d113 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x403, x397, x411):
        x404=self.conv2d113(x403)
        x412=torch.cat([x397, x404, x411], 1)
        x413=self.batchnorm2d116(x412)
        return x413

m = M().eval()
x403 = torch.randn(torch.Size([1, 192, 7, 7]))
x397 = torch.randn(torch.Size([1, 1056, 7, 7]))
x411 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x403, x397, x411)
end = time.time()
print(end-start)
