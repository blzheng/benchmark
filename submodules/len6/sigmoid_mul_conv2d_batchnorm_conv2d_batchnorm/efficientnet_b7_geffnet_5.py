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
        self.conv2d256 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d152 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d257 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d153 = BatchNorm2d(3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x764, x760):
        x765=x764.sigmoid()
        x766=operator.mul(x760, x765)
        x767=self.conv2d256(x766)
        x768=self.batchnorm2d152(x767)
        x769=self.conv2d257(x768)
        x770=self.batchnorm2d153(x769)
        return x770

m = M().eval()
x764 = torch.randn(torch.Size([1, 2304, 1, 1]))
x760 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x764, x760)
end = time.time()
print(end-start)
