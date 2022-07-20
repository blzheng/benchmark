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
        self.conv2d191 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid31 = Sigmoid()
        self.conv2d192 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d128 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x616, x613):
        x617=self.conv2d191(x616)
        x618=self.sigmoid31(x617)
        x619=operator.mul(x618, x613)
        x620=self.conv2d192(x619)
        x621=self.batchnorm2d128(x620)
        return x621

m = M().eval()
x616 = torch.randn(torch.Size([1, 96, 1, 1]))
x613 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x616, x613)
end = time.time()
print(end-start)