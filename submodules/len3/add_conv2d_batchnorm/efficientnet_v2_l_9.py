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
        self.conv2d19 = Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x69, x63):
        x70=operator.add(x69, x63)
        x71=self.conv2d19(x70)
        x72=self.batchnorm2d19(x71)
        return x72

m = M().eval()
x69 = torch.randn(torch.Size([1, 64, 56, 56]))
x63 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x69, x63)
end = time.time()
print(end-start)
