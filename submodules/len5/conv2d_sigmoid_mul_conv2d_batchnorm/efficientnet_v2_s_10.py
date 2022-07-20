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
        self.conv2d72 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d73 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x230, x227):
        x231=self.conv2d72(x230)
        x232=self.sigmoid10(x231)
        x233=operator.mul(x232, x227)
        x234=self.conv2d73(x233)
        x235=self.batchnorm2d51(x234)
        return x235

m = M().eval()
x230 = torch.randn(torch.Size([1, 40, 1, 1]))
x227 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x230, x227)
end = time.time()
print(end-start)
