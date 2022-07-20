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
        self.conv2d28 = Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)

    def forward(self, x90):
        x91=self.conv2d28(x90)
        x92=self.batchnorm2d28(x91)
        x93=self.relu19(x92)
        x94=self.conv2d29(x93)
        return x94

m = M().eval()
x90 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x90)
end = time.time()
print(end-start)
