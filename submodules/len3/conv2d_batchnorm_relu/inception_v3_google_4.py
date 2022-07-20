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
        self.conv2d4 = Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x26):
        x27=self.conv2d4(x26)
        x28=self.batchnorm2d4(x27)
        x29=torch.nn.functional.relu(x28,inplace=True)
        return x29

m = M().eval()
x26 = torch.randn(torch.Size([1, 80, 54, 54]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
