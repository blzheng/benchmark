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
        self.batchnorm2d94 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x333):
        x334=self.batchnorm2d94(x333)
        x335=self.relu94(x334)
        x336=self.conv2d94(x335)
        return x336

m = M().eval()
x333 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x333)
end = time.time()
print(end-start)
