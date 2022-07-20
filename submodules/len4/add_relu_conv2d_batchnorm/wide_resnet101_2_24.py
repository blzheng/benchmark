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
        self.relu73 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d79 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x258, x250):
        x259=operator.add(x258, x250)
        x260=self.relu73(x259)
        x261=self.conv2d79(x260)
        x262=self.batchnorm2d79(x261)
        return x262

m = M().eval()
x258 = torch.randn(torch.Size([1, 1024, 14, 14]))
x250 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x258, x250)
end = time.time()
print(end-start)
