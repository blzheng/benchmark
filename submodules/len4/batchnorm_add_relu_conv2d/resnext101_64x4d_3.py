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
        self.batchnorm2d10 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x33, x26):
        x34=self.batchnorm2d10(x33)
        x35=operator.add(x34, x26)
        x36=self.relu7(x35)
        x37=self.conv2d11(x36)
        return x37

m = M().eval()
x33 = torch.randn(torch.Size([1, 256, 56, 56]))
x26 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x33, x26)
end = time.time()
print(end-start)
