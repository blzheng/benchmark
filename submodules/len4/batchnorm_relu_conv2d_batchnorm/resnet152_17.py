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
        self.batchnorm2d28 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x92):
        x93=self.batchnorm2d28(x92)
        x94=self.relu25(x93)
        x95=self.conv2d29(x94)
        x96=self.batchnorm2d29(x95)
        return x96

m = M().eval()
x92 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
