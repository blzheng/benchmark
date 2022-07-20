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
        self.conv2d26 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(80, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x74):
        x75=self.conv2d26(x74)
        x76=self.batchnorm2d26(x75)
        return x76

m = M().eval()
x74 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x74)
end = time.time()
print(end-start)
