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
        self.conv2d8 = Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
        self.batchnorm2d7 = BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x22):
        x23=self.conv2d8(x22)
        x24=self.batchnorm2d6(x23)
        x25=self.relu4(x24)
        x26=self.conv2d9(x25)
        x27=self.batchnorm2d7(x26)
        return x27

m = M().eval()
x22 = torch.randn(torch.Size([1, 24, 28, 28]))
start = time.time()
output = m(x22)
end = time.time()
print(end-start)
