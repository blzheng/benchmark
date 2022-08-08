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
        self.relu34 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d41 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x131):
        x132=self.relu34(x131)
        x133=self.conv2d40(x132)
        x134=self.batchnorm2d40(x133)
        x135=self.relu37(x134)
        x136=self.conv2d41(x135)
        x137=self.batchnorm2d41(x136)
        x138=self.relu37(x137)
        x139=self.conv2d42(x138)
        return x139

m = M().eval()
x131 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
