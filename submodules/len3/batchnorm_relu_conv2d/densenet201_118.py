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
        self.batchnorm2d119 = BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu119 = ReLU(inplace=True)
        self.conv2d119 = Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x421):
        x422=self.batchnorm2d119(x421)
        x423=self.relu119(x422)
        x424=self.conv2d119(x423)
        return x424

m = M().eval()
x421 = torch.randn(torch.Size([1, 1536, 14, 14]))
start = time.time()
output = m(x421)
end = time.time()
print(end-start)
