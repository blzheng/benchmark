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
        self.relu97 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d102 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x332):
        x333=self.relu97(x332)
        x334=self.conv2d101(x333)
        x335=self.batchnorm2d101(x334)
        x336=self.relu97(x335)
        x337=self.conv2d102(x336)
        return x337

m = M().eval()
x332 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x332)
end = time.time()
print(end-start)
