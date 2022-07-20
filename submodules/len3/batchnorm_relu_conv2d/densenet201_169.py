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
        self.batchnorm2d170 = BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu170 = ReLU(inplace=True)
        self.conv2d170 = Conv2d(1440, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x601):
        x602=self.batchnorm2d170(x601)
        x603=self.relu170(x602)
        x604=self.conv2d170(x603)
        return x604

m = M().eval()
x601 = torch.randn(torch.Size([1, 1440, 7, 7]))
start = time.time()
output = m(x601)
end = time.time()
print(end-start)
