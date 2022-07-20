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
        self.conv2d111 = Conv2d(2112, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool2d2 = AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm2d114 = BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x395, x404):
        x396=self.conv2d111(x395)
        x397=self.avgpool2d2(x396)
        x405=torch.cat([x397, x404], 1)
        x406=self.batchnorm2d114(x405)
        return x406

m = M().eval()
x395 = torch.randn(torch.Size([1, 2112, 14, 14]))
x404 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x395, x404)
end = time.time()
print(end-start)
