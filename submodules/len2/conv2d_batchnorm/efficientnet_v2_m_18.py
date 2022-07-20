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
        self.conv2d18 = Conv2d(80, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x63):
        x64=self.conv2d18(x63)
        x65=self.batchnorm2d18(x64)
        return x65

m = M().eval()
x63 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)
