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
        self.conv2d118 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d119 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x350, x338):
        x351=self.conv2d118(x350)
        x352=self.batchnorm2d70(x351)
        x353=operator.add(x352, x338)
        x354=self.conv2d119(x353)
        x355=self.batchnorm2d71(x354)
        return x355

m = M().eval()
x350 = torch.randn(torch.Size([1, 1392, 7, 7]))
x338 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x350, x338)
end = time.time()
print(end-start)
