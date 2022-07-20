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
        self.conv2d182 = Conv2d(1824, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d183 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x537, x542):
        x543=operator.mul(x537, x542)
        x544=self.conv2d182(x543)
        x545=self.batchnorm2d108(x544)
        x546=self.conv2d183(x545)
        return x546

m = M().eval()
x537 = torch.randn(torch.Size([1, 1824, 7, 7]))
x542 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x537, x542)
end = time.time()
print(end-start)
