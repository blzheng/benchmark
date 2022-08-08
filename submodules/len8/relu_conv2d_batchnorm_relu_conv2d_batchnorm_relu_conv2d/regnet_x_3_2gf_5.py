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
        self.relu21 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d25 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x76):
        x77=self.relu21(x76)
        x78=self.conv2d24(x77)
        x79=self.batchnorm2d24(x78)
        x80=self.relu22(x79)
        x81=self.conv2d25(x80)
        x82=self.batchnorm2d25(x81)
        x83=self.relu23(x82)
        x84=self.conv2d26(x83)
        return x84

m = M().eval()
x76 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)
