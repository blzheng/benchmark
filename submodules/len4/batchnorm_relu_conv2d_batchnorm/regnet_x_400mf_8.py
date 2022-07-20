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
        self.batchnorm2d14 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x43):
        x44=self.batchnorm2d14(x43)
        x45=self.relu11(x44)
        x46=self.conv2d15(x45)
        x47=self.batchnorm2d15(x46)
        return x47

m = M().eval()
x43 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
