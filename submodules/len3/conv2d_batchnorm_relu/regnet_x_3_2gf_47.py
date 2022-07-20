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
        self.conv2d74 = Conv2d(432, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)

    def forward(self, x239):
        x242=self.conv2d74(x239)
        x243=self.batchnorm2d74(x242)
        x244=self.relu70(x243)
        return x244

m = M().eval()
x239 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)