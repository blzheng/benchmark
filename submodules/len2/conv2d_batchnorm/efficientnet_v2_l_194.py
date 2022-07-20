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
        self.conv2d302 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d194 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x971):
        x972=self.conv2d302(x971)
        x973=self.batchnorm2d194(x972)
        return x973

m = M().eval()
x971 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x971)
end = time.time()
print(end-start)
