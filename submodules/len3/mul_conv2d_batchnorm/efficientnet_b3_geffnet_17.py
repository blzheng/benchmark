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
        self.conv2d88 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x255, x260):
        x261=operator.mul(x255, x260)
        x262=self.conv2d88(x261)
        x263=self.batchnorm2d52(x262)
        return x263

m = M().eval()
x255 = torch.randn(torch.Size([1, 816, 14, 14]))
x260 = torch.randn(torch.Size([1, 816, 1, 1]))
start = time.time()
output = m(x255, x260)
end = time.time()
print(end-start)
