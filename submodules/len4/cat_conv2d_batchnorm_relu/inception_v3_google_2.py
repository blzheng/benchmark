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
        self.conv2d19 = Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x56, x62, x71, x75):
        x76=torch.cat([x56, x62, x71, x75], 1)
        x77=self.conv2d19(x76)
        x78=self.batchnorm2d19(x77)
        x79=torch.nn.functional.relu(x78,inplace=True)
        return x79

m = M().eval()
x56 = torch.randn(torch.Size([1, 64, 25, 25]))
x62 = torch.randn(torch.Size([1, 64, 25, 25]))
x71 = torch.randn(torch.Size([1, 96, 25, 25]))
x75 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x56, x62, x71, x75)
end = time.time()
print(end-start)
