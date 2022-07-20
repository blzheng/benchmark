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
        self.batchnorm2d3 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d4 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x10):
        x11=self.batchnorm2d3(x10)
        x12=self.relu2(x11)
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d4(x13)
        return x14

m = M().eval()
x10 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x10)
end = time.time()
print(end-start)
