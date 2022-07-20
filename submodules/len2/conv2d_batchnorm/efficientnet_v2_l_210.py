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
        self.conv2d328 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d210 = BatchNorm2d(3840, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1053):
        x1054=self.conv2d328(x1053)
        x1055=self.batchnorm2d210(x1054)
        return x1055

m = M().eval()
x1053 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1053)
end = time.time()
print(end-start)
