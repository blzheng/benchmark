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
        self.sigmoid3 = Sigmoid()
        self.conv2d17 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x50, x46):
        x51=self.sigmoid3(x50)
        x52=operator.mul(x51, x46)
        x53=self.conv2d17(x52)
        x54=self.batchnorm2d9(x53)
        return x54

m = M().eval()
x50 = torch.randn(torch.Size([1, 144, 1, 1]))
x46 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x50, x46)
end = time.time()
print(end-start)
