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
        self.conv2d97 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d98 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x306, x303):
        x307=self.conv2d97(x306)
        x308=self.sigmoid18(x307)
        x309=operator.mul(x308, x303)
        x310=self.conv2d98(x309)
        x311=self.batchnorm2d60(x310)
        return x311

m = M().eval()
x306 = torch.randn(torch.Size([1, 84, 1, 1]))
x303 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x306, x303)
end = time.time()
print(end-start)
