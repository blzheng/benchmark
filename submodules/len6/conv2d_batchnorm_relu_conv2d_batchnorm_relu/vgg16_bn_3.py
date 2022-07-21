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
        self.conv2d5 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d5 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d6 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)

    def forward(self, x17):
        x18=self.conv2d5(x17)
        x19=self.batchnorm2d5(x18)
        x20=self.relu5(x19)
        x21=self.conv2d6(x20)
        x22=self.batchnorm2d6(x21)
        x23=self.relu6(x22)
        return x23

m = M().eval()
x17 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)
