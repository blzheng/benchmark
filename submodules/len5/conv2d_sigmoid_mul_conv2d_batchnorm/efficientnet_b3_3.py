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
        self.conv2d17 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d18 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x50, x47):
        x51=self.conv2d17(x50)
        x52=self.sigmoid3(x51)
        x53=operator.mul(x52, x47)
        x54=self.conv2d18(x53)
        x55=self.batchnorm2d10(x54)
        return x55

m = M().eval()
x50 = torch.randn(torch.Size([1, 8, 1, 1]))
x47 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x50, x47)
end = time.time()
print(end-start)
