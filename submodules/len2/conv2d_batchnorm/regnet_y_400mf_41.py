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
        self.conv2d65 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x203):
        x204=self.conv2d65(x203)
        x205=self.batchnorm2d41(x204)
        return x205

m = M().eval()
x203 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x203)
end = time.time()
print(end-start)
