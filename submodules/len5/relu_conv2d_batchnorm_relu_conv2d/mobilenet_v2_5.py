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
        self.relu610 = ReLU6(inplace=True)
        self.conv2d16 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d16 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu611 = ReLU6(inplace=True)
        self.conv2d17 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x44):
        x45=self.relu610(x44)
        x46=self.conv2d16(x45)
        x47=self.batchnorm2d16(x46)
        x48=self.relu611(x47)
        x49=self.conv2d17(x48)
        return x49

m = M().eval()
x44 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
