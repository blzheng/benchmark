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
        self.conv2d11 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x37, x41):
        x38=self.conv2d11(x37)
        x39=self.batchnorm2d11(x38)
        x42=operator.add(x39, x41)
        x43=self.relu9(x42)
        x44=self.conv2d13(x43)
        return x44

m = M().eval()
x37 = torch.randn(torch.Size([1, 256, 14, 14]))
x41 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x37, x41)
end = time.time()
print(end-start)
