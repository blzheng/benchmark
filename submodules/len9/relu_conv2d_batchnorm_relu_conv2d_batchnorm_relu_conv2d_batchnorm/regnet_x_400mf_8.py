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
        self.relu36 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d42 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x130):
        x131=self.relu36(x130)
        x132=self.conv2d41(x131)
        x133=self.batchnorm2d41(x132)
        x134=self.relu37(x133)
        x135=self.conv2d42(x134)
        x136=self.batchnorm2d42(x135)
        x137=self.relu38(x136)
        x138=self.conv2d43(x137)
        x139=self.batchnorm2d43(x138)
        return x139

m = M().eval()
x130 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)
