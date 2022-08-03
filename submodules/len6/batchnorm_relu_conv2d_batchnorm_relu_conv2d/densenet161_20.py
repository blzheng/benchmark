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
        self.batchnorm2d43 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x155):
        x156=self.batchnorm2d43(x155)
        x157=self.relu43(x156)
        x158=self.conv2d43(x157)
        x159=self.batchnorm2d44(x158)
        x160=self.relu44(x159)
        x161=self.conv2d44(x160)
        return x161

m = M().eval()
x155 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x155)
end = time.time()
print(end-start)
