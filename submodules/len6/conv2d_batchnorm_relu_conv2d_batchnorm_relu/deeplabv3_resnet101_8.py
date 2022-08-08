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
        self.conv2d28 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d29 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x92):
        x93=self.conv2d28(x92)
        x94=self.batchnorm2d28(x93)
        x95=self.relu25(x94)
        x96=self.conv2d29(x95)
        x97=self.batchnorm2d29(x96)
        x98=self.relu25(x97)
        return x98

m = M().eval()
x92 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
