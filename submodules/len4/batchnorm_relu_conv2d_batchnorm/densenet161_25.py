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
        self.batchnorm2d53 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x190):
        x191=self.batchnorm2d53(x190)
        x192=self.relu53(x191)
        x193=self.conv2d53(x192)
        x194=self.batchnorm2d54(x193)
        return x194

m = M().eval()
x190 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
