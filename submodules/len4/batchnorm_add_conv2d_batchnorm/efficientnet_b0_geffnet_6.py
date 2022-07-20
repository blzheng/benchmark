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
        self.batchnorm2d44 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d75 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x218, x205):
        x219=self.batchnorm2d44(x218)
        x220=operator.add(x219, x205)
        x221=self.conv2d75(x220)
        x222=self.batchnorm2d45(x221)
        return x222

m = M().eval()
x218 = torch.randn(torch.Size([1, 192, 7, 7]))
x205 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x218, x205)
end = time.time()
print(end-start)
