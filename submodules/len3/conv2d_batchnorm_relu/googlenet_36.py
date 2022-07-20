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
        self.conv2d36 = Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x125):
        x135=self.conv2d36(x125)
        x136=self.batchnorm2d36(x135)
        x137=torch.nn.functional.relu(x136,inplace=True)
        return x137

m = M().eval()
x125 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
