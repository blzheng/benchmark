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
        self.relu24 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=18, bias=False)
        self.batchnorm2d29 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x88):
        x89=self.relu24(x88)
        x90=self.conv2d28(x89)
        x91=self.batchnorm2d28(x90)
        x92=self.relu25(x91)
        x93=self.conv2d29(x92)
        x94=self.batchnorm2d29(x93)
        return x94

m = M().eval()
x88 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
