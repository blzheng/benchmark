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
        self.batchnorm2d25 = BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d44 = Conv2d(88, 528, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x132):
        x133=self.batchnorm2d25(x132)
        x134=self.conv2d44(x133)
        return x134

m = M().eval()
x132 = torch.randn(torch.Size([1, 88, 14, 14]))
start = time.time()
output = m(x132)
end = time.time()
print(end-start)
