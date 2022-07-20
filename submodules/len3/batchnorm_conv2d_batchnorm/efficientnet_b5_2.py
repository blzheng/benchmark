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
        self.batchnorm2d24 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d43 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x131):
        x132=self.batchnorm2d24(x131)
        x133=self.conv2d43(x132)
        x134=self.batchnorm2d25(x133)
        return x134

m = M().eval()
x131 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
