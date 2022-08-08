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
        self.relu94 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)
        self.conv2d102 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.batchnorm2d102 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x333):
        x334=self.relu94(x333)
        x335=self.conv2d101(x334)
        x336=self.batchnorm2d101(x335)
        x337=self.relu97(x336)
        x338=self.conv2d102(x337)
        x339=self.batchnorm2d102(x338)
        x340=self.relu97(x339)
        return x340

m = M().eval()
x333 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x333)
end = time.time()
print(end-start)
