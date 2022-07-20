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
        self.conv2d85 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d86 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x266, x263):
        x267=self.conv2d85(x266)
        x268=self.sigmoid17(x267)
        x269=operator.mul(x268, x263)
        x270=self.conv2d86(x269)
        x271=self.batchnorm2d50(x270)
        return x271

m = M().eval()
x266 = torch.randn(torch.Size([1, 20, 1, 1]))
x263 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x266, x263)
end = time.time()
print(end-start)
