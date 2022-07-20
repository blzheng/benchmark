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

    def forward(self, x410):
        x411=self.conv2d124(x410)
        x412=self.batchnorm2d124(x411)
        x413=self.relu121(x412)
        return x413

m = M().eval()
x410 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x410)
end = time.time()
print(end-start)
