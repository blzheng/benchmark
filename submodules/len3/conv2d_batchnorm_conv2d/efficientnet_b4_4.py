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
        self.conv2d83 = Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d84 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x257):
        x258=self.conv2d83(x257)
        x259=self.batchnorm2d49(x258)
        x260=self.conv2d84(x259)
        return x260

m = M().eval()
x257 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x257)
end = time.time()
print(end-start)
