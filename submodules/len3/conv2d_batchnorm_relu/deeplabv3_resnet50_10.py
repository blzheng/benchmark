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
        self.conv2d16 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x53):
        x54=self.conv2d16(x53)
        x55=self.batchnorm2d16(x54)
        x56=self.relu13(x55)
        return x56

m = M().eval()
x53 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
