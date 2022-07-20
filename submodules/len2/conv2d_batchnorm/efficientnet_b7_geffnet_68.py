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
        self.conv2d116 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x348):
        x349=self.conv2d116(x348)
        x350=self.batchnorm2d68(x349)
        return x350

m = M().eval()
x348 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x348)
end = time.time()
print(end-start)
