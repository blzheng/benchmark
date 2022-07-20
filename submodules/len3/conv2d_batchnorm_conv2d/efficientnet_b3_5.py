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
        self.conv2d93 = Conv2d(816, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d94 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x287):
        x288=self.conv2d93(x287)
        x289=self.batchnorm2d55(x288)
        x290=self.conv2d94(x289)
        return x290

m = M().eval()
x287 = torch.randn(torch.Size([1, 816, 7, 7]))
start = time.time()
output = m(x287)
end = time.time()
print(end-start)
