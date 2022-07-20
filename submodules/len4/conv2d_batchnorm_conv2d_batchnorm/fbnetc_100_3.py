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
        self.conv2d39 = Conv2d(384, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d40 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x127):
        x128=self.conv2d39(x127)
        x129=self.batchnorm2d39(x128)
        x130=self.conv2d40(x129)
        x131=self.batchnorm2d40(x130)
        return x131

m = M().eval()
x127 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x127)
end = time.time()
print(end-start)
