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
        self.conv2d312 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d200 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1001):
        x1002=self.conv2d312(x1001)
        x1003=self.batchnorm2d200(x1002)
        return x1003

m = M().eval()
x1001 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1001)
end = time.time()
print(end-start)
