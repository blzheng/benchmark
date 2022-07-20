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
        self.conv2d39 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1152, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)

    def forward(self, x111):
        x112=self.conv2d39(x111)
        x113=self.batchnorm2d39(x112)
        x114=self.relu26(x113)
        x115=self.conv2d40(x114)
        return x115

m = M().eval()
x111 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)
