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
        self.conv2d127 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d128 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d86 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x409, x406):
        x410=self.conv2d127(x409)
        x411=self.sigmoid20(x410)
        x412=operator.mul(x411, x406)
        x413=self.conv2d128(x412)
        x414=self.batchnorm2d86(x413)
        return x414

m = M().eval()
x409 = torch.randn(torch.Size([1, 44, 1, 1]))
x406 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x409, x406)
end = time.time()
print(end-start)
