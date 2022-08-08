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
        self.conv2d37 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d38 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x120):
        x121=self.conv2d37(x120)
        x122=self.batchnorm2d37(x121)
        x123=self.relu25(x122)
        x124=self.conv2d38(x123)
        x125=self.batchnorm2d38(x124)
        x126=self.relu26(x125)
        return x126

m = M().eval()
x120 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
