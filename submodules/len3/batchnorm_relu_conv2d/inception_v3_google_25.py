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
        self.batchnorm2d47 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x167):
        x168=self.batchnorm2d47(x167)
        x169=torch.nn.functional.relu(x168,inplace=True)
        x170=self.conv2d48(x169)
        return x170

m = M().eval()
x167 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x167)
end = time.time()
print(end-start)
