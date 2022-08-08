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
        self.conv2d100 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d100 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x330, x324):
        x331=self.conv2d100(x330)
        x332=self.batchnorm2d100(x331)
        x333=operator.add(x332, x324)
        return x333

m = M().eval()
x330 = torch.randn(torch.Size([1, 512, 28, 28]))
x324 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x330, x324)
end = time.time()
print(end-start)
