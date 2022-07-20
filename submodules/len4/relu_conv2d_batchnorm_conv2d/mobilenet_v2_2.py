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
        self.relu67 = ReLU6(inplace=True)
        self.conv2d11 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d12 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x30):
        x31=self.relu67(x30)
        x32=self.conv2d11(x31)
        x33=self.batchnorm2d11(x32)
        x34=self.conv2d12(x33)
        return x34

m = M().eval()
x30 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
