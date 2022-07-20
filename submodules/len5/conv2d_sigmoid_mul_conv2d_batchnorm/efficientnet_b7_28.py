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
        self.conv2d140 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()
        self.conv2d141 = Conv2d(960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x440, x437):
        x441=self.conv2d140(x440)
        x442=self.sigmoid28(x441)
        x443=operator.mul(x442, x437)
        x444=self.conv2d141(x443)
        x445=self.batchnorm2d83(x444)
        return x445

m = M().eval()
x440 = torch.randn(torch.Size([1, 40, 1, 1]))
x437 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x440, x437)
end = time.time()
print(end-start)
