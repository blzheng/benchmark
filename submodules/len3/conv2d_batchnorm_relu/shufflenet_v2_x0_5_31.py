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
        self.conv2d48 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x304):
        x305=self.conv2d48(x304)
        x306=self.batchnorm2d48(x305)
        x307=self.relu31(x306)
        return x307

m = M().eval()
x304 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x304)
end = time.time()
print(end-start)
