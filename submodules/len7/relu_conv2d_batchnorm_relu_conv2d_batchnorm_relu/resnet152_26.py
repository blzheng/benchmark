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
        self.relu79 = ReLU(inplace=True)
        self.conv2d85 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d86 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x279):
        x280=self.relu79(x279)
        x281=self.conv2d85(x280)
        x282=self.batchnorm2d85(x281)
        x283=self.relu82(x282)
        x284=self.conv2d86(x283)
        x285=self.batchnorm2d86(x284)
        x286=self.relu82(x285)
        return x286

m = M().eval()
x279 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x279)
end = time.time()
print(end-start)
