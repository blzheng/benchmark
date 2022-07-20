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
        self.conv2d103 = Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x316, x311):
        x317=operator.mul(x316, x311)
        x318=self.conv2d103(x317)
        x319=self.batchnorm2d61(x318)
        return x319

m = M().eval()
x316 = torch.randn(torch.Size([1, 1248, 1, 1]))
x311 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x316, x311)
end = time.time()
print(end-start)
