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
        self.conv2d39 = Conv2d(96, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)

    def forward(self, x126):
        x127=self.conv2d39(x126)
        x128=self.batchnorm2d39(x127)
        x129=self.relu26(x128)
        x130=self.conv2d40(x129)
        return x130

m = M().eval()
x126 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x126)
end = time.time()
print(end-start)
