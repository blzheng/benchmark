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
        self.conv2d42 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu628 = ReLU6(inplace=True)
        self.conv2d43 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d43 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu629 = ReLU6(inplace=True)

    def forward(self, x120):
        x121=self.conv2d42(x120)
        x122=self.batchnorm2d42(x121)
        x123=self.relu628(x122)
        x124=self.conv2d43(x123)
        x125=self.batchnorm2d43(x124)
        x126=self.relu629(x125)
        return x126

m = M().eval()
x120 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
