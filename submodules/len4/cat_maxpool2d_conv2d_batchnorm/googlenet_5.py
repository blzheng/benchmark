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
        self.maxpool2d12 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d56 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169, x175, x181, x185):
        x186=torch.cat([x169, x175, x181, x185], 1)
        x202=self.maxpool2d12(x186)
        x203=self.conv2d56(x202)
        x204=self.batchnorm2d56(x203)
        return x204

m = M().eval()
x169 = torch.randn(torch.Size([1, 256, 7, 7]))
x175 = torch.randn(torch.Size([1, 320, 7, 7]))
x181 = torch.randn(torch.Size([1, 128, 7, 7]))
x185 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x169, x175, x181, x185)
end = time.time()
print(end-start)
