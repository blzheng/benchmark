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
        self.conv2d68 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x203, x189):
        x204=operator.add(x203, x189)
        x205=self.conv2d68(x204)
        x206=self.batchnorm2d40(x205)
        return x206

m = M().eval()
x203 = torch.randn(torch.Size([1, 72, 28, 28]))
x189 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x203, x189)
end = time.time()
print(end-start)
