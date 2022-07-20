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
        self.batchnorm2d160 = BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu160 = ReLU(inplace=True)
        self.conv2d160 = Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d161 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x566):
        x567=self.batchnorm2d160(x566)
        x568=self.relu160(x567)
        x569=self.conv2d160(x568)
        x570=self.batchnorm2d161(x569)
        return x570

m = M().eval()
x566 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x566)
end = time.time()
print(end-start)
