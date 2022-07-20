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
        self.relu5 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x27):
        x28=self.relu5(x27)
        x29=self.conv2d10(x28)
        x30=self.batchnorm2d8(x29)
        return x30

m = M().eval()
x27 = torch.randn(torch.Size([1, 88, 28, 28]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
