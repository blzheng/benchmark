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
        self.conv2d27 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x80, x88):
        x89=self.conv2d27(x80)
        x90=self.batchnorm2d27(x89)
        x91=operator.add(x88, x90)
        x92=self.relu22(x91)
        return x92

m = M().eval()
x80 = torch.randn(torch.Size([1, 512, 28, 28]))
x88 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x80, x88)
end = time.time()
print(end-start)
