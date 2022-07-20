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
        self.relu25 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d22 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x105):
        x106=self.relu25(x105)
        x107=self.conv2d34(x106)
        x108=self.batchnorm2d22(x107)
        x109=self.relu26(x108)
        return x109

m = M().eval()
x105 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
