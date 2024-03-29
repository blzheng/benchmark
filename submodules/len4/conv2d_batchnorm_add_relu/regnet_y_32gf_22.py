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
        self.conv2d99 = Conv2d(1392, 3712, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d61 = BatchNorm2d(3712, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu80 = ReLU(inplace=True)

    def forward(self, x313, x329):
        x314=self.conv2d99(x313)
        x315=self.batchnorm2d61(x314)
        x330=operator.add(x315, x329)
        x331=self.relu80(x330)
        return x331

m = M().eval()
x313 = torch.randn(torch.Size([1, 1392, 14, 14]))
x329 = torch.randn(torch.Size([1, 3712, 7, 7]))
start = time.time()
output = m(x313, x329)
end = time.time()
print(end-start)
