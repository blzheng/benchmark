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
        self.relu27 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d41 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x131):
        x132=self.relu27(x131)
        x133=self.conv2d41(x132)
        x134=self.batchnorm2d41(x133)
        return x134

m = M().eval()
x131 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
