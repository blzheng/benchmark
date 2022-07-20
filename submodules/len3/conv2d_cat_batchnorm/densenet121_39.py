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
        self.conv2d89 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d92 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x319, x313, x327):
        x320=self.conv2d89(x319)
        x328=torch.cat([x313, x320, x327], 1)
        x329=self.batchnorm2d92(x328)
        return x329

m = M().eval()
x319 = torch.randn(torch.Size([1, 128, 7, 7]))
x313 = torch.randn(torch.Size([1, 512, 7, 7]))
x327 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x319, x313, x327)
end = time.time()
print(end-start)
