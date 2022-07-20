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
        self.conv2d57 = Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x185):
        x186=self.conv2d57(x185)
        x187=self.batchnorm2d57(x186)
        return x187

m = M().eval()
x185 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x185)
end = time.time()
print(end-start)
