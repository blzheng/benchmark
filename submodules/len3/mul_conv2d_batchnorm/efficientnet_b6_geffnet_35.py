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
        self.conv2d177 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x522, x527):
        x528=operator.mul(x522, x527)
        x529=self.conv2d177(x528)
        x530=self.batchnorm2d105(x529)
        return x530

m = M().eval()
x522 = torch.randn(torch.Size([1, 2064, 7, 7]))
x527 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x522, x527)
end = time.time()
print(end-start)
