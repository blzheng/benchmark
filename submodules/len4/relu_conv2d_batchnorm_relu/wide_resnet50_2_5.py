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
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)

    def forward(self, x35):
        x36=self.relu7(x35)
        x37=self.conv2d11(x36)
        x38=self.batchnorm2d11(x37)
        x39=self.relu10(x38)
        return x39

m = M().eval()
x35 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
