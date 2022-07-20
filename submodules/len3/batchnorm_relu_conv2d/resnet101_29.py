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
        self.batchnorm2d47 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x154):
        x155=self.batchnorm2d47(x154)
        x156=self.relu43(x155)
        x157=self.conv2d48(x156)
        return x157

m = M().eval()
x154 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x154)
end = time.time()
print(end-start)
