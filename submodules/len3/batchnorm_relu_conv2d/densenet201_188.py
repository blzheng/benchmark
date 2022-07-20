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
        self.batchnorm2d189 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu189 = ReLU(inplace=True)
        self.conv2d189 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x667):
        x668=self.batchnorm2d189(x667)
        x669=self.relu189(x668)
        x670=self.conv2d189(x669)
        return x670

m = M().eval()
x667 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x667)
end = time.time()
print(end-start)
