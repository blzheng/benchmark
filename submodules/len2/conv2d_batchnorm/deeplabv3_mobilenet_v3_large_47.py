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
        self.conv2d63 = Conv2d(960, 256, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), bias=False)
        self.batchnorm2d47 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x183):
        x187=self.conv2d63(x183)
        x188=self.batchnorm2d47(x187)
        return x188

m = M().eval()
x183 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
