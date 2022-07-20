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
        self.relu32 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232, bias=False)
        self.batchnorm2d50 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x323):
        x324=self.relu32(x323)
        x325=self.conv2d50(x324)
        x326=self.batchnorm2d50(x325)
        x327=self.conv2d51(x326)
        x328=self.batchnorm2d51(x327)
        return x328

m = M().eval()
x323 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x323)
end = time.time()
print(end-start)
