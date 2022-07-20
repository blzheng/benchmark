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
        self.batchnorm2d24 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87):
        x88=self.batchnorm2d24(x87)
        x89=self.relu24(x88)
        x90=self.conv2d24(x89)
        x91=self.batchnorm2d25(x90)
        return x91

m = M().eval()
x87 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
