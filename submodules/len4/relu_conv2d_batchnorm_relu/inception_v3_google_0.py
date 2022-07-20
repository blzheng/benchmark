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
        self.conv2d1 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x15):
        x16=torch.nn.functional.relu(x15,inplace=True)
        x17=self.conv2d1(x16)
        x18=self.batchnorm2d1(x17)
        x19=torch.nn.functional.relu(x18,inplace=True)
        return x19

m = M().eval()
x15 = torch.randn(torch.Size([1, 32, 111, 111]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
