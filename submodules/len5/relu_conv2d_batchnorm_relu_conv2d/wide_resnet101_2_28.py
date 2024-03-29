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
        self.relu43 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x152):
        x153=self.relu43(x152)
        x154=self.conv2d47(x153)
        x155=self.batchnorm2d47(x154)
        x156=self.relu43(x155)
        x157=self.conv2d48(x156)
        return x157

m = M().eval()
x152 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
