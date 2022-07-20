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
        self.relu13 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x56):
        x57=self.relu13(x56)
        x58=self.conv2d20(x57)
        x59=self.batchnorm2d20(x58)
        return x59

m = M().eval()
x56 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x56)
end = time.time()
print(end-start)
