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
        self.batchnorm2d15 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d16 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x48):
        x49=self.batchnorm2d15(x48)
        x50=self.relu13(x49)
        x51=self.conv2d16(x50)
        x52=self.batchnorm2d16(x51)
        x53=self.relu14(x52)
        x54=self.conv2d17(x53)
        x55=self.batchnorm2d17(x54)
        return x55

m = M().eval()
x48 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
