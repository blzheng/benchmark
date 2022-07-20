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
        self.relu121 = ReLU(inplace=True)
        self.conv2d125 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d125 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x412):
        x413=self.relu121(x412)
        x414=self.conv2d125(x413)
        x415=self.batchnorm2d125(x414)
        return x415

m = M().eval()
x412 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
