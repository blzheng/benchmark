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
        self.batchnorm2d144 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu144 = ReLU(inplace=True)
        self.conv2d144 = Conv2d(1280, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d145 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu145 = ReLU(inplace=True)
        self.conv2d145 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x510):
        x511=self.batchnorm2d144(x510)
        x512=self.relu144(x511)
        x513=self.conv2d144(x512)
        x514=self.batchnorm2d145(x513)
        x515=self.relu145(x514)
        x516=self.conv2d145(x515)
        return x516

m = M().eval()
x510 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x510)
end = time.time()
print(end-start)
