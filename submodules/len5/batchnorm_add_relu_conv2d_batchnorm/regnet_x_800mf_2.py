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
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x16, x25):
        x17=self.batchnorm2d5(x16)
        x26=operator.add(x17, x25)
        x27=self.relu6(x26)
        x28=self.conv2d9(x27)
        x29=self.batchnorm2d9(x28)
        return x29

m = M().eval()
x16 = torch.randn(torch.Size([1, 128, 28, 28]))
x25 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x16, x25)
end = time.time()
print(end-start)
