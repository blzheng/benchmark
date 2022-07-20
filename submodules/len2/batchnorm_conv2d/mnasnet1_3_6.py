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
        self.batchnorm2d50 = BatchNorm2d(416, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(416, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x145):
        x146=self.batchnorm2d50(x145)
        x147=self.conv2d51(x146)
        return x147

m = M().eval()
x145 = torch.randn(torch.Size([1, 416, 7, 7]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
