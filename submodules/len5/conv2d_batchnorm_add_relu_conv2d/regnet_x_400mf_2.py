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
        self.conv2d5 = Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d5 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x15, x25):
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d5(x16)
        x26=operator.add(x17, x25)
        x27=self.relu6(x26)
        x28=self.conv2d9(x27)
        return x28

m = M().eval()
x15 = torch.randn(torch.Size([1, 32, 56, 56]))
x25 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x15, x25)
end = time.time()
print(end-start)
