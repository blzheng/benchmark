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
        self.batchnorm2d89 = BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu89 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu90 = ReLU(inplace=True)

    def forward(self, x316):
        x317=self.batchnorm2d89(x316)
        x318=self.relu89(x317)
        x319=self.conv2d89(x318)
        x320=self.batchnorm2d90(x319)
        x321=self.relu90(x320)
        return x321

m = M().eval()
x316 = torch.randn(torch.Size([1, 1584, 14, 14]))
start = time.time()
output = m(x316)
end = time.time()
print(end-start)
