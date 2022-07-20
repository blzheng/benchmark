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
        self.conv2d184 = Conv2d(1664, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d185 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu185 = ReLU(inplace=True)
        self.conv2d185 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x652):
        x653=self.conv2d184(x652)
        x654=self.batchnorm2d185(x653)
        x655=self.relu185(x654)
        x656=self.conv2d185(x655)
        return x656

m = M().eval()
x652 = torch.randn(torch.Size([1, 1664, 7, 7]))
start = time.time()
output = m(x652)
end = time.time()
print(end-start)
