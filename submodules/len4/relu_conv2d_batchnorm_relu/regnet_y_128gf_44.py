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
        self.relu93 = ReLU(inplace=True)
        self.conv2d120 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d74 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)

    def forward(self, x379):
        x380=self.relu93(x379)
        x381=self.conv2d120(x380)
        x382=self.batchnorm2d74(x381)
        x383=self.relu94(x382)
        return x383

m = M().eval()
x379 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x379)
end = time.time()
print(end-start)
