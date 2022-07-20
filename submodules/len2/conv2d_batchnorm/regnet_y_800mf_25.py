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
        self.conv2d39 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x121):
        x122=self.conv2d39(x121)
        x123=self.batchnorm2d25(x122)
        return x123

m = M().eval()
x121 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
