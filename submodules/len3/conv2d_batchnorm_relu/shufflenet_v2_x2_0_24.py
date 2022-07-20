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
        self.conv2d37 = Conv2d(244, 244, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(244, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)

    def forward(self, x236):
        x237=self.conv2d37(x236)
        x238=self.batchnorm2d37(x237)
        x239=self.relu24(x238)
        return x239

m = M().eval()
x236 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
