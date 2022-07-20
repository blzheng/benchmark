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
        self.relu27 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x107):
        x108=self.relu27(x107)
        x109=self.conv2d32(x108)
        x110=self.batchnorm2d32(x109)
        return x110

m = M().eval()
x107 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x107)
end = time.time()
print(end-start)
