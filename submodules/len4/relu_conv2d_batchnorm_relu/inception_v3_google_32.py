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
        self.conv2d62 = Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d62 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x214):
        x215=torch.nn.functional.relu(x214,inplace=True)
        x216=self.conv2d62(x215)
        x217=self.batchnorm2d62(x216)
        x218=torch.nn.functional.relu(x217,inplace=True)
        return x218

m = M().eval()
x214 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x214)
end = time.time()
print(end-start)
