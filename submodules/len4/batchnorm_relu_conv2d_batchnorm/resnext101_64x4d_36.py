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
        self.batchnorm2d58 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d59 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x191):
        x192=self.batchnorm2d58(x191)
        x193=self.relu55(x192)
        x194=self.conv2d59(x193)
        x195=self.batchnorm2d59(x194)
        return x195

m = M().eval()
x191 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x191)
end = time.time()
print(end-start)
