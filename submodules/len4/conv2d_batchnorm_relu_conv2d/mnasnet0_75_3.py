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
        self.conv2d4 = Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d4 = BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x11):
        x12=self.conv2d4(x11)
        x13=self.batchnorm2d4(x12)
        x14=self.relu3(x13)
        x15=self.conv2d5(x14)
        return x15

m = M().eval()
x11 = torch.randn(torch.Size([1, 48, 112, 112]))
start = time.time()
output = m(x11)
end = time.time()
print(end-start)
