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
        self.conv2d88 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x316):
        x317=self.conv2d88(x316)
        x318=self.batchnorm2d89(x317)
        return x318

m = M().eval()
x316 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x316)
end = time.time()
print(end-start)
