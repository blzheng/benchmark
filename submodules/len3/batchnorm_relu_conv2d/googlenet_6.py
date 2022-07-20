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
        self.batchnorm2d18 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x75):
        x76=self.batchnorm2d18(x75)
        x77=torch.nn.functional.relu(x76,inplace=True)
        x78=self.conv2d19(x77)
        return x78

m = M().eval()
x75 = torch.randn(torch.Size([1, 16, 14, 14]))
start = time.time()
output = m(x75)
end = time.time()
print(end-start)
