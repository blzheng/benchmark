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
        self.conv2d109 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x323, x309):
        x324=operator.add(x323, x309)
        x325=self.conv2d109(x324)
        x326=self.batchnorm2d65(x325)
        return x326

m = M().eval()
x323 = torch.randn(torch.Size([1, 160, 14, 14]))
x309 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x323, x309)
end = time.time()
print(end-start)
