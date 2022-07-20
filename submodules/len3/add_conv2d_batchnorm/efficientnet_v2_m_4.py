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
        self.conv2d10 = Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x36, x30):
        x37=operator.add(x36, x30)
        x38=self.conv2d10(x37)
        x39=self.batchnorm2d10(x38)
        return x39

m = M().eval()
x36 = torch.randn(torch.Size([1, 48, 56, 56]))
x30 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x36, x30)
end = time.time()
print(end-start)
