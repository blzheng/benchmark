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
        self.conv2d41 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d41 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)

    def forward(self, x132):
        x133=self.conv2d41(x132)
        x134=self.batchnorm2d41(x133)
        x135=self.relu28(x134)
        return x135

m = M().eval()
x132 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x132)
end = time.time()
print(end-start)
