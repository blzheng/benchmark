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
        self.sigmoid2 = Sigmoid()
        self.conv2d33 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x105, x101):
        x106=self.sigmoid2(x105)
        x107=operator.mul(x106, x101)
        x108=self.conv2d33(x107)
        x109=self.batchnorm2d27(x108)
        return x109

m = M().eval()
x105 = torch.randn(torch.Size([1, 512, 1, 1]))
x101 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x105, x101)
end = time.time()
print(end-start)
