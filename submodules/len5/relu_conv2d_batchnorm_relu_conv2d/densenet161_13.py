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
        self.relu28 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x102):
        x103=self.relu28(x102)
        x104=self.conv2d28(x103)
        x105=self.batchnorm2d29(x104)
        x106=self.relu29(x105)
        x107=self.conv2d29(x106)
        return x107

m = M().eval()
x102 = torch.randn(torch.Size([1, 528, 28, 28]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
