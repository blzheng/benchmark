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
        self.conv2d307 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d197 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d308 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x987):
        x988=self.conv2d307(x987)
        x989=self.batchnorm2d197(x988)
        x990=self.conv2d308(x989)
        return x990

m = M().eval()
x987 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x987)
end = time.time()
print(end-start)
