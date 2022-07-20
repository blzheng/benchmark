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
        self.conv2d297 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d191 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x954, x949):
        x955=operator.mul(x954, x949)
        x956=self.conv2d297(x955)
        x957=self.batchnorm2d191(x956)
        return x957

m = M().eval()
x954 = torch.randn(torch.Size([1, 2304, 1, 1]))
x949 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x954, x949)
end = time.time()
print(end-start)
