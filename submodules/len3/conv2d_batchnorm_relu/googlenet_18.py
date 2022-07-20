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
        self.conv2d18 = Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x65):
        x75=self.conv2d18(x65)
        x76=self.batchnorm2d18(x75)
        x77=torch.nn.functional.relu(x76,inplace=True)
        return x77

m = M().eval()
x65 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x65)
end = time.time()
print(end-start)
