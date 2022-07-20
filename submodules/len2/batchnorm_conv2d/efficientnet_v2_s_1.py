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
        self.batchnorm2d12 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x43):
        x44=self.batchnorm2d12(x43)
        x45=self.conv2d13(x44)
        return x45

m = M().eval()
x43 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
