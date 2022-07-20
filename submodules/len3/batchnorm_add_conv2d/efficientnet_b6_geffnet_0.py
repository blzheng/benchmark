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
        self.batchnorm2d24 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d43 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x128, x115):
        x129=self.batchnorm2d24(x128)
        x130=operator.add(x129, x115)
        x131=self.conv2d43(x130)
        return x131

m = M().eval()
x128 = torch.randn(torch.Size([1, 40, 56, 56]))
x115 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x128, x115)
end = time.time()
print(end-start)
