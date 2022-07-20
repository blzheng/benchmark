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
        self.sigmoid17 = Sigmoid()
        self.conv2d108 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x341, x337):
        x342=self.sigmoid17(x341)
        x343=operator.mul(x342, x337)
        x344=self.conv2d108(x343)
        x345=self.batchnorm2d72(x344)
        return x345

m = M().eval()
x341 = torch.randn(torch.Size([1, 1536, 1, 1]))
x337 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x341, x337)
end = time.time()
print(end-start)
