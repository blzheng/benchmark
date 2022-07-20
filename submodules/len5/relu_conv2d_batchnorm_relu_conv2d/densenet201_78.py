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
        self.relu160 = ReLU(inplace=True)
        self.conv2d160 = Conv2d(1280, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d161 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu161 = ReLU(inplace=True)
        self.conv2d161 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x567):
        x568=self.relu160(x567)
        x569=self.conv2d160(x568)
        x570=self.batchnorm2d161(x569)
        x571=self.relu161(x570)
        x572=self.conv2d161(x571)
        return x572

m = M().eval()
x567 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x567)
end = time.time()
print(end-start)
