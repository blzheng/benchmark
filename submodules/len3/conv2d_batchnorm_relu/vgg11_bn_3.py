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
        self.conv2d3 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)

    def forward(self, x11):
        x12=self.conv2d3(x11)
        x13=self.batchnorm2d3(x12)
        x14=self.relu3(x13)
        return x14

m = M().eval()
x11 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x11)
end = time.time()
print(end-start)
