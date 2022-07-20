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
        self.conv2d8 = Conv2d(48, 104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(104, 104, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=13, bias=False)
        self.batchnorm2d7 = BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x21):
        x24=self.conv2d8(x21)
        x25=self.batchnorm2d6(x24)
        x26=self.relu5(x25)
        x27=self.conv2d9(x26)
        x28=self.batchnorm2d7(x27)
        return x28

m = M().eval()
x21 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
