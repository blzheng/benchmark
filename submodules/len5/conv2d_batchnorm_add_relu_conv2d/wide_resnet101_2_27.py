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
        self.conv2d78 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x256, x250):
        x257=self.conv2d78(x256)
        x258=self.batchnorm2d78(x257)
        x259=operator.add(x258, x250)
        x260=self.relu73(x259)
        x261=self.conv2d79(x260)
        return x261

m = M().eval()
x256 = torch.randn(torch.Size([1, 512, 14, 14]))
x250 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x256, x250)
end = time.time()
print(end-start)
