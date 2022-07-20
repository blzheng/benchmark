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
        self.conv2d54 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d55 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x175):
        x176=self.conv2d54(x175)
        x177=self.batchnorm2d54(x176)
        x178=self.relu36(x177)
        x179=self.conv2d55(x178)
        x180=self.batchnorm2d55(x179)
        return x180

m = M().eval()
x175 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)