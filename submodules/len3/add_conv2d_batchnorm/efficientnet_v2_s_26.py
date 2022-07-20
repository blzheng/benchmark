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
        self.conv2d129 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x410, x395):
        x411=operator.add(x410, x395)
        x412=self.conv2d129(x411)
        x413=self.batchnorm2d85(x412)
        return x413

m = M().eval()
x410 = torch.randn(torch.Size([1, 256, 7, 7]))
x395 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x410, x395)
end = time.time()
print(end-start)
