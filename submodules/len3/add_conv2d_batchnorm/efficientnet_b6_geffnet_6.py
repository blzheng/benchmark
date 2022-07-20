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
        self.conv2d43 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x129, x115):
        x130=operator.add(x129, x115)
        x131=self.conv2d43(x130)
        x132=self.batchnorm2d25(x131)
        return x132

m = M().eval()
x129 = torch.randn(torch.Size([1, 40, 56, 56]))
x115 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x129, x115)
end = time.time()
print(end-start)
