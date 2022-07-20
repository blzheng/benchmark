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
        self.conv2d13 = Conv2d(96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
        self.batchnorm2d13 = BatchNorm2d(96, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(96, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x37):
        x38=self.conv2d13(x37)
        x39=self.batchnorm2d13(x38)
        x40=self.relu9(x39)
        x41=self.conv2d14(x40)
        return x41

m = M().eval()
x37 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
