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
        self.conv2d23 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)

    def forward(self, x76, x70):
        x77=self.conv2d23(x76)
        x78=self.batchnorm2d23(x77)
        x79=operator.add(x78, x70)
        x80=self.relu19(x79)
        return x80

m = M().eval()
x76 = torch.randn(torch.Size([1, 128, 28, 28]))
x70 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x76, x70)
end = time.time()
print(end-start)
