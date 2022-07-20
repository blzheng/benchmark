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
        self.batchnorm2d31 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x159):
        x160=self.batchnorm2d31(x159)
        x161=self.conv2d54(x160)
        return x161

m = M().eval()
x159 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x159)
end = time.time()
print(end-start)
