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
        self.conv2d78 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(768, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x241, x226):
        x242=operator.add(x241, x226)
        x243=self.conv2d78(x242)
        x244=self.batchnorm2d46(x243)
        return x244

m = M().eval()
x241 = torch.randn(torch.Size([1, 128, 14, 14]))
x226 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x241, x226)
end = time.time()
print(end-start)
