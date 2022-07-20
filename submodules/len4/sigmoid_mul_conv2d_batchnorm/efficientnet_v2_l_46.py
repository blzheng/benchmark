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
        self.sigmoid46 = Sigmoid()
        self.conv2d267 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d173 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x857, x853):
        x858=self.sigmoid46(x857)
        x859=operator.mul(x858, x853)
        x860=self.conv2d267(x859)
        x861=self.batchnorm2d173(x860)
        return x861

m = M().eval()
x857 = torch.randn(torch.Size([1, 2304, 1, 1]))
x853 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x857, x853)
end = time.time()
print(end-start)
