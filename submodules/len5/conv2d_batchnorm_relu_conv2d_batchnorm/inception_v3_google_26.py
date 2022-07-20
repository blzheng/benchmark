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
        self.conv2d51 = Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d52 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x177):
        x181=self.conv2d51(x177)
        x182=self.batchnorm2d51(x181)
        x183=torch.nn.functional.relu(x182,inplace=True)
        x184=self.conv2d52(x183)
        x185=self.batchnorm2d52(x184)
        return x185

m = M().eval()
x177 = torch.randn(torch.Size([1, 768, 12, 12]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
