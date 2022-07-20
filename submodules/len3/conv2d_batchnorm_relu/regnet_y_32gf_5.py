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
        self.conv2d13 = Conv2d(232, 696, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)

    def forward(self, x37):
        x40=self.conv2d13(x37)
        x41=self.batchnorm2d9(x40)
        x42=self.relu9(x41)
        return x42

m = M().eval()
x37 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
