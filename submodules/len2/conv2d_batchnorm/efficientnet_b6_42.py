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
        self.conv2d72 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x224):
        x225=self.conv2d72(x224)
        x226=self.batchnorm2d42(x225)
        return x226

m = M().eval()
x224 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
