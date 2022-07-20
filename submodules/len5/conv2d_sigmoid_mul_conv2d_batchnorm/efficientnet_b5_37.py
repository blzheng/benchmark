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
        self.conv2d186 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()
        self.conv2d187 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(512, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x581, x578):
        x582=self.conv2d186(x581)
        x583=self.sigmoid37(x582)
        x584=operator.mul(x583, x578)
        x585=self.conv2d187(x584)
        x586=self.batchnorm2d111(x585)
        return x586

m = M().eval()
x581 = torch.randn(torch.Size([1, 128, 1, 1]))
x578 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x581, x578)
end = time.time()
print(end-start)
