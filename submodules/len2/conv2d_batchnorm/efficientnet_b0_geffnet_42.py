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
        self.conv2d70 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x205):
        x206=self.conv2d70(x205)
        x207=self.batchnorm2d42(x206)
        return x207

m = M().eval()
x205 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
