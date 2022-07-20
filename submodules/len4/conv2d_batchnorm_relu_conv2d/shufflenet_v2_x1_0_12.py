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
        self.conv2d43 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(232, 232, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=232, bias=False)

    def forward(self, x272):
        x278=self.conv2d43(x272)
        x279=self.batchnorm2d43(x278)
        x280=self.relu28(x279)
        x281=self.conv2d44(x280)
        return x281

m = M().eval()
x272 = torch.randn(torch.Size([1, 232, 14, 14]))
start = time.time()
output = m(x272)
end = time.time()
print(end-start)
