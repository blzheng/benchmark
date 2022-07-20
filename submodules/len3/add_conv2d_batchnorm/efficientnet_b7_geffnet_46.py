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
        self.conv2d267 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d159 = BatchNorm2d(3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x797, x783):
        x798=operator.add(x797, x783)
        x799=self.conv2d267(x798)
        x800=self.batchnorm2d159(x799)
        return x800

m = M().eval()
x797 = torch.randn(torch.Size([1, 640, 7, 7]))
x783 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x797, x783)
end = time.time()
print(end-start)
