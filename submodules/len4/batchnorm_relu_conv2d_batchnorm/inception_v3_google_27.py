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
        self.batchnorm2d52 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d53 = Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d53 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x184):
        x185=self.batchnorm2d52(x184)
        x186=torch.nn.functional.relu(x185,inplace=True)
        x187=self.conv2d53(x186)
        x188=self.batchnorm2d53(x187)
        return x188

m = M().eval()
x184 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
