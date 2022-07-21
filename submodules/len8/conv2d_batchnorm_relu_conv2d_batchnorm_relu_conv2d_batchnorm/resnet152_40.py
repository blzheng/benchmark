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
        self.conv2d124 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d124 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu121 = ReLU(inplace=True)
        self.conv2d125 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d125 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d126 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x410):
        x411=self.conv2d124(x410)
        x412=self.batchnorm2d124(x411)
        x413=self.relu121(x412)
        x414=self.conv2d125(x413)
        x415=self.batchnorm2d125(x414)
        x416=self.relu121(x415)
        x417=self.conv2d126(x416)
        x418=self.batchnorm2d126(x417)
        return x418

m = M().eval()
x410 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x410)
end = time.time()
print(end-start)
