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
        self.relu11 = ReLU()
        self.conv2d16 = Conv2d(58, 696, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d17 = Conv2d(696, 696, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x47, x45):
        x48=self.relu11(x47)
        x49=self.conv2d16(x48)
        x50=self.sigmoid2(x49)
        x51=operator.mul(x50, x45)
        x52=self.conv2d17(x51)
        x53=self.batchnorm2d11(x52)
        return x53

m = M().eval()
x47 = torch.randn(torch.Size([1, 58, 1, 1]))
x45 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x47, x45)
end = time.time()
print(end-start)
