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
        self.conv2d69 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(1344, 2520, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x225, x219):
        x226=self.conv2d69(x225)
        x227=self.batchnorm2d69(x226)
        x228=operator.add(x219, x227)
        x229=self.relu66(x228)
        x230=self.conv2d70(x229)
        return x230

m = M().eval()
x225 = torch.randn(torch.Size([1, 1344, 14, 14]))
x219 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x225, x219)
end = time.time()
print(end-start)
