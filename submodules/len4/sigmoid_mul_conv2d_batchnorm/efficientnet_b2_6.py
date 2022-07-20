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
        self.sigmoid6 = Sigmoid()
        self.conv2d33 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x97, x93):
        x98=self.sigmoid6(x97)
        x99=operator.mul(x98, x93)
        x100=self.conv2d33(x99)
        x101=self.batchnorm2d19(x100)
        return x101

m = M().eval()
x97 = torch.randn(torch.Size([1, 288, 1, 1]))
x93 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x97, x93)
end = time.time()
print(end-start)
