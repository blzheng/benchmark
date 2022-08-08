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
        self.conv2d36 = Conv2d(24, 384, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d37 = Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d38 = Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x126, x123):
        x127=self.conv2d36(x126)
        x128=self.sigmoid0(x127)
        x129=operator.mul(x128, x123)
        x130=self.conv2d37(x129)
        x131=self.batchnorm2d35(x130)
        x132=self.conv2d38(x131)
        return x132

m = M().eval()
x126 = torch.randn(torch.Size([1, 24, 1, 1]))
x123 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x126, x123)
end = time.time()
print(end-start)
