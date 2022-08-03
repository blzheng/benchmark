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
        self.conv2d41 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d42 = Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d43 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x127, x124):
        x128=self.conv2d41(x127)
        x129=self.sigmoid8(x128)
        x130=operator.mul(x129, x124)
        x131=self.conv2d42(x130)
        x132=self.batchnorm2d24(x131)
        x133=self.conv2d43(x132)
        return x133

m = M().eval()
x127 = torch.randn(torch.Size([1, 10, 1, 1]))
x124 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x127, x124)
end = time.time()
print(end-start)
