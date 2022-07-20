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
        self.batchnorm2d51 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu51 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)

    def forward(self, x183):
        x184=self.batchnorm2d51(x183)
        x185=self.relu51(x184)
        x186=self.conv2d51(x185)
        x187=self.batchnorm2d52(x186)
        x188=self.relu52(x187)
        return x188

m = M().eval()
x183 = torch.randn(torch.Size([1, 448, 14, 14]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
