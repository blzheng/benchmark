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
        self.relu94 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x332, x324):
        x333=operator.add(x332, x324)
        x334=self.relu94(x333)
        x335=self.conv2d101(x334)
        x336=self.batchnorm2d101(x335)
        return x336

m = M().eval()
x332 = torch.randn(torch.Size([1, 2048, 28, 28]))
x324 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x332, x324)
end = time.time()
print(end-start)
