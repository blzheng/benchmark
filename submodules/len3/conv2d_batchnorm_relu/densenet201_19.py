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
        self.conv2d39 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x143):
        x144=self.conv2d39(x143)
        x145=self.batchnorm2d40(x144)
        x146=self.relu40(x145)
        return x146

m = M().eval()
x143 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
