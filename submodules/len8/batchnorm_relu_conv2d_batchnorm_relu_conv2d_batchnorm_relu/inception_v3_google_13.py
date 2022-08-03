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
        self.batchnorm2d51 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d52 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d53 = Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d53 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x181):
        x182=self.batchnorm2d51(x181)
        x183=torch.nn.functional.relu(x182,inplace=True)
        x184=self.conv2d52(x183)
        x185=self.batchnorm2d52(x184)
        x186=torch.nn.functional.relu(x185,inplace=True)
        x187=self.conv2d53(x186)
        x188=self.batchnorm2d53(x187)
        x189=torch.nn.functional.relu(x188,inplace=True)
        return x189

m = M().eval()
x181 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x181)
end = time.time()
print(end-start)
