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
        self.conv2d46 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d28 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x134):
        x135=self.conv2d46(x134)
        x136=self.batchnorm2d28(x135)
        return x136

m = M().eval()
x134 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x134)
end = time.time()
print(end-start)
