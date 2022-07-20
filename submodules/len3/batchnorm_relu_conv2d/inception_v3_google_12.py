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
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x103):
        x104=self.batchnorm2d27(x103)
        x105=torch.nn.functional.relu(x104,inplace=True)
        x106=self.conv2d28(x105)
        return x106

m = M().eval()
x103 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
