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
        self.conv2d5 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x30):
        x31=self.conv2d5(x30)
        x32=self.batchnorm2d5(x31)
        x33=torch.nn.functional.relu(x32,inplace=True)
        return x33

m = M().eval()
x30 = torch.randn(torch.Size([1, 192, 25, 25]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
