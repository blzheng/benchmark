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
        self.conv2d21 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x59):
        x60=self.conv2d21(x59)
        x61=self.batchnorm2d21(x60)
        return x61

m = M().eval()
x59 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x59)
end = time.time()
print(end-start)
