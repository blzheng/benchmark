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
        self.batchnorm2d15 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x72, x87):
        x73=self.batchnorm2d15(x72)
        x88=operator.add(x73, x87)
        x89=self.relu20(x88)
        x90=self.conv2d29(x89)
        x91=self.batchnorm2d19(x90)
        return x91

m = M().eval()
x72 = torch.randn(torch.Size([1, 208, 14, 14]))
x87 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x72, x87)
end = time.time()
print(end-start)
