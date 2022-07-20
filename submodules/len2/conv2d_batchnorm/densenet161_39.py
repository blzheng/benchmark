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
        self.conv2d79 = Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x283):
        x284=self.conv2d79(x283)
        x285=self.batchnorm2d80(x284)
        return x285

m = M().eval()
x283 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x283)
end = time.time()
print(end-start)
