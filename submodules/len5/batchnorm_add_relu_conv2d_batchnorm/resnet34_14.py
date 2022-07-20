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
        self.batchnorm2d28 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x96, x92):
        x97=self.batchnorm2d28(x96)
        x98=operator.add(x97, x92)
        x99=self.relu25(x98)
        x100=self.conv2d29(x99)
        x101=self.batchnorm2d29(x100)
        return x101

m = M().eval()
x96 = torch.randn(torch.Size([1, 256, 14, 14]))
x92 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x96, x92)
end = time.time()
print(end-start)
