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
        self.batchnorm2d18 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x56, x49):
        x57=self.batchnorm2d18(x56)
        x58=operator.add(x49, x57)
        x59=self.relu15(x58)
        x60=self.conv2d19(x59)
        return x60

m = M().eval()
x56 = torch.randn(torch.Size([1, 160, 14, 14]))
x49 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x56, x49)
end = time.time()
print(end-start)
