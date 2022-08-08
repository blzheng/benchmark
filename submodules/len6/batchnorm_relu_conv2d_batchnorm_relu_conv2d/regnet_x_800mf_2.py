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
        self.batchnorm2d9 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        self.batchnorm2d10 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x28):
        x29=self.batchnorm2d9(x28)
        x30=self.relu7(x29)
        x31=self.conv2d10(x30)
        x32=self.batchnorm2d10(x31)
        x33=self.relu8(x32)
        x34=self.conv2d11(x33)
        return x34

m = M().eval()
x28 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
