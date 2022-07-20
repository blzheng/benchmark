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
        self.conv2d55 = Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x197):
        x198=torch.nn.functional.relu(x197,inplace=True)
        x199=self.conv2d55(x198)
        x200=self.batchnorm2d55(x199)
        x201=torch.nn.functional.relu(x200,inplace=True)
        return x201

m = M().eval()
x197 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x197)
end = time.time()
print(end-start)
