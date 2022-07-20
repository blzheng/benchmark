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
        self.conv2d53 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(432, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x164):
        x165=self.conv2d53(x164)
        x166=self.batchnorm2d31(x165)
        return x166

m = M().eval()
x164 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
