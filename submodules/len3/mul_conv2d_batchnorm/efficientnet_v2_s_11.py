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
        self.conv2d78 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x248, x243):
        x249=operator.mul(x248, x243)
        x250=self.conv2d78(x249)
        x251=self.batchnorm2d54(x250)
        return x251

m = M().eval()
x248 = torch.randn(torch.Size([1, 960, 1, 1]))
x243 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x248, x243)
end = time.time()
print(end-start)
