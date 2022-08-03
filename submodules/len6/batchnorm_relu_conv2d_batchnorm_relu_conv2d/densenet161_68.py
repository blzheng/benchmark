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
        self.batchnorm2d140 = BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu140 = ReLU(inplace=True)
        self.conv2d140 = Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d141 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu141 = ReLU(inplace=True)
        self.conv2d141 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x496):
        x497=self.batchnorm2d140(x496)
        x498=self.relu140(x497)
        x499=self.conv2d140(x498)
        x500=self.batchnorm2d141(x499)
        x501=self.relu141(x500)
        x502=self.conv2d141(x501)
        return x502

m = M().eval()
x496 = torch.randn(torch.Size([1, 1728, 7, 7]))
start = time.time()
output = m(x496)
end = time.time()
print(end-start)
