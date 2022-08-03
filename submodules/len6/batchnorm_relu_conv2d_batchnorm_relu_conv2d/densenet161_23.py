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
        self.batchnorm2d49 = BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x176):
        x177=self.batchnorm2d49(x176)
        x178=self.relu49(x177)
        x179=self.conv2d49(x178)
        x180=self.batchnorm2d50(x179)
        x181=self.relu50(x180)
        x182=self.conv2d50(x181)
        return x182

m = M().eval()
x176 = torch.randn(torch.Size([1, 624, 14, 14]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
