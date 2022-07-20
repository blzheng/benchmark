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
        self.conv2d30 = Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640, bias=False)
        self.batchnorm2d28 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x101):
        x102=self.conv2d30(x101)
        x103=self.batchnorm2d28(x102)
        return x103

m = M().eval()
x101 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
