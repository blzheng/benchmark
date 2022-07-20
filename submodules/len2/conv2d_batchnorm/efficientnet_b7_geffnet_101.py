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
        self.conv2d171 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x512):
        x513=self.conv2d171(x512)
        x514=self.batchnorm2d101(x513)
        return x514

m = M().eval()
x512 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x512)
end = time.time()
print(end-start)
