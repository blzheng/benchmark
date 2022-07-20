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
        self.relu121 = ReLU(inplace=True)
        self.conv2d121 = Conv2d(1568, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d122 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x429):
        x430=self.relu121(x429)
        x431=self.conv2d121(x430)
        x432=self.batchnorm2d122(x431)
        return x432

m = M().eval()
x429 = torch.randn(torch.Size([1, 1568, 14, 14]))
start = time.time()
output = m(x429)
end = time.time()
print(end-start)
