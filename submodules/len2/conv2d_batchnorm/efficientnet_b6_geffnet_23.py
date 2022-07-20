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
        self.conv2d39 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d23 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x118):
        x119=self.conv2d39(x118)
        x120=self.batchnorm2d23(x119)
        return x120

m = M().eval()
x118 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x118)
end = time.time()
print(end-start)
