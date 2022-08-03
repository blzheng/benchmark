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
        self.batchnorm2d128 = BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu128 = ReLU(inplace=True)
        self.conv2d128 = Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d129 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu129 = ReLU(inplace=True)
        self.conv2d129 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x454):
        x455=self.batchnorm2d128(x454)
        x456=self.relu128(x455)
        x457=self.conv2d128(x456)
        x458=self.batchnorm2d129(x457)
        x459=self.relu129(x458)
        x460=self.conv2d129(x459)
        return x460

m = M().eval()
x454 = torch.randn(torch.Size([1, 1440, 7, 7]))
start = time.time()
output = m(x454)
end = time.time()
print(end-start)
