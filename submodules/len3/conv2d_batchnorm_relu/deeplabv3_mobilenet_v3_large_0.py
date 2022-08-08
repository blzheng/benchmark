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
        self.conv2d1 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d1 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)

    def forward(self, x5):
        x6=self.conv2d1(x5)
        x7=self.batchnorm2d1(x6)
        x8=self.relu0(x7)
        return x8

m = M().eval()
x5 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x5)
end = time.time()
print(end-start)
