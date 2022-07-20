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

    def forward(self, x760, x765):
        x766=operator.mul(x760, x765)
        x767=self.conv2d256(x766)
        x768=self.batchnorm2d152(x767)
        x769=self.conv2d257(x768)
        return x769

m = M().eval()
x760 = torch.randn(torch.Size([1, 2304, 7, 7]))
x765 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x760, x765)
end = time.time()
print(end-start)
