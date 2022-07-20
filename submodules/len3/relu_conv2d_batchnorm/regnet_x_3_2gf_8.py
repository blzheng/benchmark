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
        self.relu8 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x32):
        x33=self.relu8(x32)
        x34=self.conv2d11(x33)
        x35=self.batchnorm2d11(x34)
        return x35

m = M().eval()
x32 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
