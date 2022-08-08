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
        self.conv2d117 = Conv2d(864, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d118 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x348, x344):
        x349=x348.sigmoid()
        x350=operator.mul(x344, x349)
        x351=self.conv2d117(x350)
        x352=self.batchnorm2d69(x351)
        x353=self.conv2d118(x352)
        x354=self.batchnorm2d70(x353)
        return x354

m = M().eval()
x348 = torch.randn(torch.Size([1, 864, 1, 1]))
x344 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x348, x344)
end = time.time()
print(end-start)
