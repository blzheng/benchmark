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
        self.batchnorm2d126 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu126 = ReLU(inplace=True)
        self.conv2d126 = Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d127 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu127 = ReLU(inplace=True)

    def forward(self, x447):
        x448=self.batchnorm2d126(x447)
        x449=self.relu126(x448)
        x450=self.conv2d126(x449)
        x451=self.batchnorm2d127(x450)
        x452=self.relu127(x451)
        return x452

m = M().eval()
x447 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x447)
end = time.time()
print(end-start)
