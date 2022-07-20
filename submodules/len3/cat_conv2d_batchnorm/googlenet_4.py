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
        self.conv2d33 = Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x108, x114, x120, x124):
        x125=torch.cat([x108, x114, x120, x124], 1)
        x126=self.conv2d33(x125)
        x127=self.batchnorm2d33(x126)
        return x127

m = M().eval()
x108 = torch.randn(torch.Size([1, 128, 14, 14]))
x114 = torch.randn(torch.Size([1, 256, 14, 14]))
x120 = torch.randn(torch.Size([1, 64, 14, 14]))
x124 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x108, x114, x120, x124)
end = time.time()
print(end-start)
