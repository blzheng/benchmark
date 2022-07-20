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
        self.conv2d39 = Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x129, x114):
        x130=operator.add(x129, x114)
        x131=self.conv2d39(x130)
        x132=self.batchnorm2d33(x131)
        return x132

m = M().eval()
x129 = torch.randn(torch.Size([1, 160, 14, 14]))
x114 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x129, x114)
end = time.time()
print(end-start)
