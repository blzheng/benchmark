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
        self.batchnorm2d58 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x193):
        x194=self.batchnorm2d58(x193)
        x195=self.relu55(x194)
        x196=self.conv2d59(x195)
        return x196

m = M().eval()
x193 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x193)
end = time.time()
print(end-start)
