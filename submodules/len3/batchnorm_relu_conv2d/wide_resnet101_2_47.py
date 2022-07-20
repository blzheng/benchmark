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
        self.batchnorm2d74 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x244):
        x245=self.batchnorm2d74(x244)
        x246=self.relu70(x245)
        x247=self.conv2d75(x246)
        return x247

m = M().eval()
x244 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x244)
end = time.time()
print(end-start)
