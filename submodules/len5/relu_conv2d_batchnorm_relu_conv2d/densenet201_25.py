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
        self.relu53 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x191):
        x192=self.relu53(x191)
        x193=self.conv2d53(x192)
        x194=self.batchnorm2d54(x193)
        x195=self.relu54(x194)
        x196=self.conv2d54(x195)
        return x196

m = M().eval()
x191 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x191)
end = time.time()
print(end-start)
