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
        self.sigmoid6 = Sigmoid()
        self.conv2d37 = Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x113, x109, x103):
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        x116=self.conv2d37(x115)
        x117=self.batchnorm2d23(x116)
        x118=operator.add(x103, x117)
        return x118

m = M().eval()
x113 = torch.randn(torch.Size([1, 216, 1, 1]))
x109 = torch.randn(torch.Size([1, 216, 28, 28]))
x103 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x113, x109, x103)
end = time.time()
print(end-start)
