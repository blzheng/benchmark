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
        self.conv2d46 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d47 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x152):
        x153=self.conv2d46(x152)
        x154=self.batchnorm2d46(x153)
        x155=self.relu43(x154)
        x156=self.conv2d47(x155)
        x157=self.batchnorm2d47(x156)
        x158=self.relu43(x157)
        x159=self.conv2d48(x158)
        return x159

m = M().eval()
x152 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
