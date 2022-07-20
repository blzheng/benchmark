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
        self.conv2d43 = Conv2d(352, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x272):
        x278=self.conv2d43(x272)
        x279=self.batchnorm2d43(x278)
        return x279

m = M().eval()
x272 = torch.randn(torch.Size([1, 352, 14, 14]))
start = time.time()
output = m(x272)
end = time.time()
print(end-start)
