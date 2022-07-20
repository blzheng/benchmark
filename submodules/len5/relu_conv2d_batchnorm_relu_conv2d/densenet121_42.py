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
        self.relu88 = ReLU(inplace=True)
        self.conv2d88 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu89 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x315):
        x316=self.relu88(x315)
        x317=self.conv2d88(x316)
        x318=self.batchnorm2d89(x317)
        x319=self.relu89(x318)
        x320=self.conv2d89(x319)
        return x320

m = M().eval()
x315 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x315)
end = time.time()
print(end-start)
