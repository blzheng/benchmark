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
        self.batchnorm2d40 = BatchNorm2d(1488, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(1488, 248, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x115):
        x116=self.batchnorm2d40(x115)
        x117=self.relu27(x116)
        x118=self.conv2d41(x117)
        return x118

m = M().eval()
x115 = torch.randn(torch.Size([1, 1488, 7, 7]))
start = time.time()
output = m(x115)
end = time.time()
print(end-start)
