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
        self.conv2d172 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()
        self.conv2d173 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x551, x548):
        x552=self.conv2d172(x551)
        x553=self.sigmoid29(x552)
        x554=operator.mul(x553, x548)
        x555=self.conv2d173(x554)
        x556=self.batchnorm2d113(x555)
        return x556

m = M().eval()
x551 = torch.randn(torch.Size([1, 76, 1, 1]))
x548 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x551, x548)
end = time.time()
print(end-start)
