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
        self.conv2d108 = Conv2d(1248, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d109 = Conv2d(352, 2112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x320):
        x321=self.conv2d108(x320)
        x322=self.batchnorm2d64(x321)
        x323=self.conv2d109(x322)
        return x323

m = M().eval()
x320 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x320)
end = time.time()
print(end-start)
