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
        self.relu40 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.batchnorm2d48 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x153):
        x154=self.relu40(x153)
        x155=self.conv2d47(x154)
        x156=self.batchnorm2d47(x155)
        x157=self.relu43(x156)
        x158=self.conv2d48(x157)
        x159=self.batchnorm2d48(x158)
        x160=self.relu43(x159)
        x161=self.conv2d49(x160)
        x162=self.batchnorm2d49(x161)
        return x162

m = M().eval()
x153 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
