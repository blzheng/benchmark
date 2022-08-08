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
        self.conv2d96 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x318, x322):
        x319=self.conv2d96(x318)
        x320=self.batchnorm2d96(x319)
        x323=operator.add(x320, x322)
        return x323

m = M().eval()
x318 = torch.randn(torch.Size([1, 512, 28, 28]))
x322 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x318, x322)
end = time.time()
print(end-start)
