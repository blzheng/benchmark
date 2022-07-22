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
        self.conv2d24 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d25 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d26 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78):
        x79=self.conv2d24(x78)
        x80=self.batchnorm2d24(x79)
        x81=self.relu22(x80)
        x82=self.conv2d25(x81)
        x83=self.batchnorm2d25(x82)
        x84=self.relu22(x83)
        x85=self.conv2d26(x84)
        x86=self.batchnorm2d26(x85)
        return x86

m = M().eval()
x78 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
