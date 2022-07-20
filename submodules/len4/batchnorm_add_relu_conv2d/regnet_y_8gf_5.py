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
        self.batchnorm2d14 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(448, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x68, x55):
        x69=self.batchnorm2d14(x68)
        x70=operator.add(x55, x69)
        x71=self.relu16(x70)
        x72=self.conv2d23(x71)
        return x72

m = M().eval()
x68 = torch.randn(torch.Size([1, 448, 28, 28]))
x55 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x68, x55)
end = time.time()
print(end-start)
