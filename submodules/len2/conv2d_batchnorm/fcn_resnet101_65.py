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
        self.conv2d65 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d65 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x215):
        x216=self.conv2d65(x215)
        x217=self.batchnorm2d65(x216)
        return x217

m = M().eval()
x215 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x215)
end = time.time()
print(end-start)
