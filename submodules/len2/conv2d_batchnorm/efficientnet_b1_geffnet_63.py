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
        self.conv2d105 = Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
        self.batchnorm2d63 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x311):
        x312=self.conv2d105(x311)
        x313=self.batchnorm2d63(x312)
        return x313

m = M().eval()
x311 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x311)
end = time.time()
print(end-start)
