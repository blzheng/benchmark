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
        self.conv2d57 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d58 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x186, x180):
        x187=self.conv2d57(x186)
        x188=self.batchnorm2d57(x187)
        x189=operator.add(x188, x180)
        x190=self.relu52(x189)
        x191=self.conv2d58(x190)
        x192=self.batchnorm2d58(x191)
        return x192

m = M().eval()
x186 = torch.randn(torch.Size([1, 512, 14, 14]))
x180 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x186, x180)
end = time.time()
print(end-start)
