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
        self.conv2d85 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)

    def forward(self, x280):
        x281=self.conv2d85(x280)
        x282=self.batchnorm2d85(x281)
        x283=self.relu82(x282)
        x284=self.conv2d86(x283)
        return x284

m = M().eval()
x280 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
