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
        self.sigmoid9 = Sigmoid()
        self.conv2d48 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x143, x139):
        x144=self.sigmoid9(x143)
        x145=operator.mul(x144, x139)
        x146=self.conv2d48(x145)
        x147=self.batchnorm2d28(x146)
        return x147

m = M().eval()
x143 = torch.randn(torch.Size([1, 576, 1, 1]))
x139 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x143, x139)
end = time.time()
print(end-start)
