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
        self.batchnorm2d120 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu120 = ReLU(inplace=True)
        self.conv2d120 = Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d121 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x426):
        x427=self.batchnorm2d120(x426)
        x428=self.relu120(x427)
        x429=self.conv2d120(x428)
        x430=self.batchnorm2d121(x429)
        return x430

m = M().eval()
x426 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x426)
end = time.time()
print(end-start)
