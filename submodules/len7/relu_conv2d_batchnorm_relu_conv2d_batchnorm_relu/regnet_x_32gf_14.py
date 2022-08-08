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
        self.relu51 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        self.batchnorm2d56 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)

    def forward(self, x178):
        x179=self.relu51(x178)
        x180=self.conv2d55(x179)
        x181=self.batchnorm2d55(x180)
        x182=self.relu52(x181)
        x183=self.conv2d56(x182)
        x184=self.batchnorm2d56(x183)
        x185=self.relu53(x184)
        return x185

m = M().eval()
x178 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
