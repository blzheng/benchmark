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
        self.batchnorm2d131 = BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu131 = ReLU(inplace=True)
        self.conv2d131 = Conv2d(1728, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x463):
        x464=self.batchnorm2d131(x463)
        x465=self.relu131(x464)
        x466=self.conv2d131(x465)
        return x466

m = M().eval()
x463 = torch.randn(torch.Size([1, 1728, 14, 14]))
start = time.time()
output = m(x463)
end = time.time()
print(end-start)
