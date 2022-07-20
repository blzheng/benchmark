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
        self.conv2d25 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x85):
        x86=self.conv2d25(x85)
        x87=self.batchnorm2d25(x86)
        x88=self.relu23(x87)
        x89=self.conv2d26(x88)
        x90=self.batchnorm2d26(x89)
        return x90

m = M().eval()
x85 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
