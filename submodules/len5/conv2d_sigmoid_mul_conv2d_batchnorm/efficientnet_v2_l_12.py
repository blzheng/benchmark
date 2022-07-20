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
        self.conv2d96 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d97 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x314, x311):
        x315=self.conv2d96(x314)
        x316=self.sigmoid12(x315)
        x317=operator.mul(x316, x311)
        x318=self.conv2d97(x317)
        x319=self.batchnorm2d71(x318)
        return x319

m = M().eval()
x314 = torch.randn(torch.Size([1, 56, 1, 1]))
x311 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x314, x311)
end = time.time()
print(end-start)
