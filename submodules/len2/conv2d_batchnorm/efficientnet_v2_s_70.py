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
        self.conv2d104 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x331):
        x332=self.conv2d104(x331)
        x333=self.batchnorm2d70(x332)
        return x333

m = M().eval()
x331 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x331)
end = time.time()
print(end-start)
