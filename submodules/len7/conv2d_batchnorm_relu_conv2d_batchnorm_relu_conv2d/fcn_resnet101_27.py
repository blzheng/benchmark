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
        self.conv2d85 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d86 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d87 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x282):
        x283=self.conv2d85(x282)
        x284=self.batchnorm2d85(x283)
        x285=self.relu82(x284)
        x286=self.conv2d86(x285)
        x287=self.batchnorm2d86(x286)
        x288=self.relu82(x287)
        x289=self.conv2d87(x288)
        return x289

m = M().eval()
x282 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x282)
end = time.time()
print(end-start)
