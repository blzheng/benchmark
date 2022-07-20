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
        self.conv2d97 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d97 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x310):
        x319=self.conv2d97(x310)
        x320=self.batchnorm2d97(x319)
        return x320

m = M().eval()
x310 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x310)
end = time.time()
print(end-start)
