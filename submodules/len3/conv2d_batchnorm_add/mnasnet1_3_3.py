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
        self.conv2d32 = Conv2d(624, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(128, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x92, x102):
        x93=self.conv2d32(x92)
        x94=self.batchnorm2d32(x93)
        x103=operator.add(x102, x94)
        return x103

m = M().eval()
x92 = torch.randn(torch.Size([1, 624, 14, 14]))
x102 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x92, x102)
end = time.time()
print(end-start)
