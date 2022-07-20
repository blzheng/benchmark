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
        self.batchnorm2d194 = BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu194 = ReLU(inplace=True)
        self.conv2d194 = Conv2d(1824, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d195 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x685):
        x686=self.batchnorm2d194(x685)
        x687=self.relu194(x686)
        x688=self.conv2d194(x687)
        x689=self.batchnorm2d195(x688)
        return x689

m = M().eval()
x685 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x685)
end = time.time()
print(end-start)
