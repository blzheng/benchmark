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
        self.batchnorm2d90 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x297, x290):
        x298=self.batchnorm2d90(x297)
        x299=operator.add(x298, x290)
        x300=self.relu85(x299)
        x301=self.conv2d91(x300)
        return x301

m = M().eval()
x297 = torch.randn(torch.Size([1, 1024, 14, 14]))
x290 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x297, x290)
end = time.time()
print(end-start)
