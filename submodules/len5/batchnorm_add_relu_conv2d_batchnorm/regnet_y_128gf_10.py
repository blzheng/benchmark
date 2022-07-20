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
        self.batchnorm2d29 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(1056, 2904, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d30 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x148, x135):
        x149=self.batchnorm2d29(x148)
        x150=operator.add(x135, x149)
        x151=self.relu36(x150)
        x152=self.conv2d48(x151)
        x153=self.batchnorm2d30(x152)
        return x153

m = M().eval()
x148 = torch.randn(torch.Size([1, 1056, 28, 28]))
x135 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x148, x135)
end = time.time()
print(end-start)
