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
        self.batchnorm2d7 = BatchNorm2d(122, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d8 = Conv2d(122, 122, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x35):
        x36=self.batchnorm2d7(x35)
        x37=self.conv2d8(x36)
        return x37

m = M().eval()
x35 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
