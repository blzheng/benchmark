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
        self.conv2d148 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d88 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x440):
        x441=self.conv2d148(x440)
        x442=self.batchnorm2d88(x441)
        return x442

m = M().eval()
x440 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x440)
end = time.time()
print(end-start)
