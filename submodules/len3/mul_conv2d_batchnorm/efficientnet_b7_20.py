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
        self.conv2d101 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x314, x309):
        x315=operator.mul(x314, x309)
        x316=self.conv2d101(x315)
        x317=self.batchnorm2d59(x316)
        return x317

m = M().eval()
x314 = torch.randn(torch.Size([1, 960, 1, 1]))
x309 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x314, x309)
end = time.time()
print(end-start)