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
        self.conv2d94 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x288, x273):
        x289=operator.add(x288, x273)
        x290=self.conv2d94(x289)
        x291=self.batchnorm2d56(x290)
        return x291

m = M().eval()
x288 = torch.randn(torch.Size([1, 192, 7, 7]))
x273 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x288, x273)
end = time.time()
print(end-start)