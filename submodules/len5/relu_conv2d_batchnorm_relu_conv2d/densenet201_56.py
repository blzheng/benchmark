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
        self.relu115 = ReLU(inplace=True)
        self.conv2d115 = Conv2d(1472, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu116 = ReLU(inplace=True)
        self.conv2d116 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x408):
        x409=self.relu115(x408)
        x410=self.conv2d115(x409)
        x411=self.batchnorm2d116(x410)
        x412=self.relu116(x411)
        x413=self.conv2d116(x412)
        return x413

m = M().eval()
x408 = torch.randn(torch.Size([1, 1472, 14, 14]))
start = time.time()
output = m(x408)
end = time.time()
print(end-start)
