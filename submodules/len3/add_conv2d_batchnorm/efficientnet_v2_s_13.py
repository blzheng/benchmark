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
        self.conv2d59 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x188, x173):
        x189=operator.add(x188, x173)
        x190=self.conv2d59(x189)
        x191=self.batchnorm2d43(x190)
        return x191

m = M().eval()
x188 = torch.randn(torch.Size([1, 160, 14, 14]))
x173 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x188, x173)
end = time.time()
print(end-start)
