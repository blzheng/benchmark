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
        self.batchnorm2d48 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x157, x150):
        x158=self.batchnorm2d48(x157)
        x159=operator.add(x158, x150)
        x160=self.relu43(x159)
        x161=self.conv2d49(x160)
        return x161

m = M().eval()
x157 = torch.randn(torch.Size([1, 1024, 14, 14]))
x150 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x157, x150)
end = time.time()
print(end-start)
