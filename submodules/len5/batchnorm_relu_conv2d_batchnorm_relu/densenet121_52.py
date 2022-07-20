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
        self.batchnorm2d108 = BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu108 = ReLU(inplace=True)
        self.conv2d108 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d109 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)

    def forward(self, x384):
        x385=self.batchnorm2d108(x384)
        x386=self.relu108(x385)
        x387=self.conv2d108(x386)
        x388=self.batchnorm2d109(x387)
        x389=self.relu109(x388)
        return x389

m = M().eval()
x384 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x384)
end = time.time()
print(end-start)
