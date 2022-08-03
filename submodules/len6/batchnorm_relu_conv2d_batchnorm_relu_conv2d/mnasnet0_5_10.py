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
        self.batchnorm2d30 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d31 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(240, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x87):
        x88=self.batchnorm2d30(x87)
        x89=self.relu20(x88)
        x90=self.conv2d31(x89)
        x91=self.batchnorm2d31(x90)
        x92=self.relu21(x91)
        x93=self.conv2d32(x92)
        return x93

m = M().eval()
x87 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
