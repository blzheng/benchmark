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
        self.relu52 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d44 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)

    def forward(self, x216):
        x217=self.relu52(x216)
        x218=self.conv2d69(x217)
        x219=self.batchnorm2d43(x218)
        x220=self.relu53(x219)
        x221=self.conv2d70(x220)
        x222=self.batchnorm2d44(x221)
        x223=self.relu54(x222)
        return x223

m = M().eval()
x216 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x216)
end = time.time()
print(end-start)
