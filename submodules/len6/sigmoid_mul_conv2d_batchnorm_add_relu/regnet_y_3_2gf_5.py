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
        self.sigmoid5 = Sigmoid()
        self.conv2d32 = Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)

    def forward(self, x97, x93, x87):
        x98=self.sigmoid5(x97)
        x99=operator.mul(x98, x93)
        x100=self.conv2d32(x99)
        x101=self.batchnorm2d20(x100)
        x102=operator.add(x87, x101)
        x103=self.relu24(x102)
        return x103

m = M().eval()
x97 = torch.randn(torch.Size([1, 216, 1, 1]))
x93 = torch.randn(torch.Size([1, 216, 28, 28]))
x87 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x97, x93, x87)
end = time.time()
print(end-start)
