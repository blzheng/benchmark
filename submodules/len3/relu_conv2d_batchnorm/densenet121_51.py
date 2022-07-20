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
        self.relu106 = ReLU(inplace=True)
        self.conv2d106 = Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x378):
        x379=self.relu106(x378)
        x380=self.conv2d106(x379)
        x381=self.batchnorm2d107(x380)
        return x381

m = M().eval()
x378 = torch.randn(torch.Size([1, 800, 7, 7]))
start = time.time()
output = m(x378)
end = time.time()
print(end-start)
