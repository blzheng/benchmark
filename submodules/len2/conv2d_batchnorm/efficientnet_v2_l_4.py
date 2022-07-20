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
        self.conv2d4 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x18):
        x19=self.conv2d4(x18)
        x20=self.batchnorm2d4(x19)
        return x20

m = M().eval()
x18 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
