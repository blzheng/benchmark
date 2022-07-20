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
        self.conv2d23 = Conv2d(104, 208, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d15 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x71, x87):
        x72=self.conv2d23(x71)
        x73=self.batchnorm2d15(x72)
        x88=operator.add(x73, x87)
        return x88

m = M().eval()
x71 = torch.randn(torch.Size([1, 104, 28, 28]))
x87 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x71, x87)
end = time.time()
print(end-start)
