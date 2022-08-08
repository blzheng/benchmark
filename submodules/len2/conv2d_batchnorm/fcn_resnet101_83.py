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
        self.conv2d83 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d83 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x275):
        x276=self.conv2d83(x275)
        x277=self.batchnorm2d83(x276)
        return x277

m = M().eval()
x275 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
