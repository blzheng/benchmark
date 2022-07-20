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
        self.conv2d221 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d131 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x661):
        x662=self.conv2d221(x661)
        x663=self.batchnorm2d131(x662)
        return x663

m = M().eval()
x661 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x661)
end = time.time()
print(end-start)
