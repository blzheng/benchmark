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
        self.conv2d6 = Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=8, bias=False)

    def forward(self, x15):
        x18=self.conv2d6(x15)
        x19=self.batchnorm2d6(x18)
        x20=self.relu4(x19)
        x21=self.conv2d7(x20)
        return x21

m = M().eval()
x15 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
