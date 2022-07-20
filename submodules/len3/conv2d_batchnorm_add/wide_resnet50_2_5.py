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
        self.conv2d14 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d14 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x36, x44):
        x45=self.conv2d14(x36)
        x46=self.batchnorm2d14(x45)
        x47=operator.add(x44, x46)
        return x47

m = M().eval()
x36 = torch.randn(torch.Size([1, 256, 56, 56]))
x44 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x36, x44)
end = time.time()
print(end-start)
